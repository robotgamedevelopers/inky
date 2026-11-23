#!/usr/bin/env python3
"""Run bounded safety checks and require each negative control to fail correctly."""
import argparse
import pathlib
import platform
import re
import subprocess
import sys
import time

RUNS = [
    ("BlockLayout3", "BlockLayout", None),
    ("BlockLayout4", "BlockLayout", None),
    ("StaleEntry", "BlockLayout", "GenerationMatches"),
    ("WindowOffByOne", "BlockLayout", "WindowCovered"),
    ("SpanTooSmall", "BlockLayout", "SpanPinned"),
    ("StartReleased", "BlockLayout", "StartRetained"),
    ("SpanPins512", "SpanPins", None),
    ("SpanPins17", "SpanPins", "Covered"),
]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--jar", type=pathlib.Path, default=pathlib.Path("/opt/tla/tla2tools.jar"))
parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("tlc-results"))
args = parser.parse_args()
if platform.system() != "Linux":
    sys.exit("Run this suite on a Linux bench, not the engine's MacBook.")
jar = args.jar.resolve(strict=True)
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=True)
root = pathlib.Path(__file__).resolve().parent
print("run | generated | distinct | wall seconds | result", flush=True)
for name, module, expected in RUNS:
    started = time.monotonic()
    result = subprocess.run(
        ["java", "-XX:+UseParallelGC", "-Xmx4g", "-cp", str(jar), "tlc2.TLC",
         "-workers", "1", "-metadir", str(output / (name + "-states")),
         "-config", str(root / (name + ".cfg")), str(root / (module + ".tla"))],
        cwd=output, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    elapsed = time.monotonic() - started
    (output / (name + ".log")).write_text(result.stdout)
    counts = re.search(r"([\d,]+) states generated, ([\d,]+) distinct states found", result.stdout)
    errors = re.findall(r"^Error: (.+)$", result.stdout, re.MULTILINE)
    if expected:
        ok = (result.returncode != 0 and errors == [
            f"Invariant {expected} is violated.", "The behavior up to this point is:"
        ] and "State 2:" in result.stdout)
    else:
        ok = (result.returncode == 0 and not errors
              and "Model checking completed. No error has been found." in result.stdout)
    ok = ok and counts is not None and "Finished in " in result.stdout
    if not ok:
        sys.exit(f"Unexpected result for {name} (exit {result.returncode}); inspect {output / (name + '.log')}")
    print(f"{name} | {counts[1]} | {counts[2]} | {elapsed:.3f} | "
          + (f"expected {expected} violation" if expected else "PASS"), flush=True)
