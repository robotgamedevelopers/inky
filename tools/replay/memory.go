package main

import (
	"errors"
	"fmt"
	"math"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
)

// SystemMemorySample captures independently measured host physical RAM and available headroom.
type SystemMemorySample struct {
	TotalBytes     uint64
	AvailableBytes uint64
	FreeBytes      uint64
}

func checkedMulUint64(a, b uint64) (uint64, error) {
	if a == 0 || b == 0 {
		return 0, nil
	}
	if a > math.MaxUint64/b {
		return 0, errors.New("uint64 multiplication overflow")
	}
	return a * b, nil
}

// ParsePSRssKiB parses `ps -o rss= -p <pid>` output (KiB) into bytes.
func ParsePSRssKiB(raw string) (uint64, error) {
	fields := strings.Fields(raw)
	if len(fields) != 1 {
		return 0, fmt.Errorf("unexpected ps rss output %q", strings.TrimSpace(raw))
	}
	kib, err := strconv.ParseUint(fields[0], 10, 64)
	if err != nil || kib == 0 {
		return 0, fmt.Errorf("invalid ps rss value %q", fields[0])
	}
	return checkedMulUint64(kib, 1024)
}

// ParseSysctlMemsize parses `sysctl -n hw.memsize` output into bytes.
func ParseSysctlMemsize(raw string) (uint64, error) {
	s := strings.TrimSpace(raw)
	if s == "" {
		return 0, errors.New("empty hw.memsize output")
	}
	bytes, err := strconv.ParseUint(s, 10, 64)
	if err != nil || bytes == 0 {
		return 0, fmt.Errorf("invalid hw.memsize value %q", s)
	}
	return bytes, nil
}

// ParseVMStat parses macOS `vm_stat` output into free bytes and reclaimable+free headroom bytes.
func ParseVMStat(raw string) (freeBytes uint64, availableBytes uint64, err error) {
	var pageSize uint64
	var freePages, speculativePages, inactivePages, purgeablePages uint64
	var sawFree bool

	for _, line := range strings.Split(raw, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		if strings.Contains(line, "page size of ") {
			idx := strings.Index(line, "page size of ")
			rest := line[idx+len("page size of "):]
			fields := strings.Fields(rest)
			if len(fields) >= 1 {
				n, parseErr := strconv.ParseUint(fields[0], 10, 64)
				if parseErr == nil && n > 0 {
					pageSize = n
				}
			}
			continue
		}
		key, val, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		val = strings.TrimSuffix(strings.TrimSpace(val), ".")
		count, parseErr := strconv.ParseUint(val, 10, 64)
		if parseErr != nil {
			continue
		}
		switch strings.TrimSpace(key) {
		case "Pages free":
			freePages = count
			sawFree = true
		case "Pages speculative":
			speculativePages = count
		case "Pages inactive":
			inactivePages = count
		case "Pages purgeable":
			purgeablePages = count
		}
	}

	if pageSize == 0 {
		return 0, 0, errors.New("vm_stat output missing page size")
	}
	if !sawFree {
		return 0, 0, errors.New("vm_stat output missing Pages free")
	}
	freeTotal := freePages + speculativePages
	availTotal := freeTotal + inactivePages + purgeablePages
	freeBytes, err = checkedMulUint64(freeTotal, pageSize)
	if err != nil {
		return 0, 0, err
	}
	availableBytes, err = checkedMulUint64(availTotal, pageSize)
	if err != nil {
		return 0, 0, err
	}
	return freeBytes, availableBytes, nil
}

// ParseProcMeminfo parses Linux `/proc/meminfo` into total, available, and free bytes.
func ParseProcMeminfo(raw string) (SystemMemorySample, error) {
	var totalKiB, availKiB, freeKiB, buffersKiB, cachedKiB uint64
	var sawTotal, sawAvail bool

	for _, line := range strings.Split(raw, "\n") {
		key, rest, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		fields := strings.Fields(rest)
		if len(fields) == 0 {
			continue
		}
		val, err := strconv.ParseUint(fields[0], 10, 64)
		if err != nil {
			continue
		}
		switch strings.TrimSpace(key) {
		case "MemTotal":
			totalKiB = val
			sawTotal = true
		case "MemAvailable":
			availKiB = val
			sawAvail = true
		case "MemFree":
			freeKiB = val
		case "Buffers":
			buffersKiB = val
		case "Cached":
			cachedKiB = val
		}
	}
	if !sawTotal || totalKiB == 0 {
		return SystemMemorySample{}, errors.New("/proc/meminfo missing MemTotal")
	}
	if !sawAvail {
		availKiB = freeKiB + buffersKiB + cachedKiB
	}
	totalB, err := checkedMulUint64(totalKiB, 1024)
	if err != nil {
		return SystemMemorySample{}, err
	}
	availB, err := checkedMulUint64(availKiB, 1024)
	if err != nil {
		return SystemMemorySample{}, err
	}
	freeB, err := checkedMulUint64(freeKiB, 1024)
	if err != nil {
		return SystemMemorySample{}, err
	}
	return SystemMemorySample{
		TotalBytes:     totalB,
		AvailableBytes: availB,
		FreeBytes:      freeB,
	}, nil
}

// SampleProcessRSS reads the current resident set size (in bytes) of pid via `ps`.
func SampleProcessRSS(pid int) (uint64, error) {
	if pid <= 0 {
		return 0, fmt.Errorf("invalid pid %d", pid)
	}
	out, err := exec.Command("ps", "-o", "rss=", "-p", strconv.Itoa(pid)).Output()
	if err != nil {
		return 0, err
	}
	return ParsePSRssKiB(string(out))
}

// SampleSystemMemory independently samples host physical memory and headroom via OS telemetry.
func SampleSystemMemory() (SystemMemorySample, error) {
	switch runtime.GOOS {
	case "darwin":
		memOut, err := exec.Command("sysctl", "-n", "hw.memsize").Output()
		if err != nil {
			return SystemMemorySample{}, fmt.Errorf("sysctl hw.memsize: %w", err)
		}
		totalBytes, err := ParseSysctlMemsize(string(memOut))
		if err != nil {
			return SystemMemorySample{}, err
		}
		vmOut, err := exec.Command("vm_stat").Output()
		if err != nil {
			return SystemMemorySample{}, fmt.Errorf("vm_stat: %w", err)
		}
		freeBytes, availBytes, err := ParseVMStat(string(vmOut))
		if err != nil {
			return SystemMemorySample{}, err
		}
		return SystemMemorySample{
			TotalBytes:     totalBytes,
			AvailableBytes: availBytes,
			FreeBytes:      freeBytes,
		}, nil
	case "linux":
		data, err := os.ReadFile("/proc/meminfo")
		if err != nil {
			return SystemMemorySample{}, err
		}
		return ParseProcMeminfo(string(data))
	default:
		return SystemMemorySample{}, fmt.Errorf("unsupported OS %q for system memory sampling", runtime.GOOS)
	}
}
