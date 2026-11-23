[![ci](https://github.com/robotgamedevelopers/inky/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/robotgamedevelopers/inky/actions/workflows/ci.yml)

Fast local inference for **Inkling 2 Mini** on Apple silicon.

With **inky**, you can:

* Run Thinking Machines' Inkling 2 Mini on an M5 Mac, from the LM Studio GGUF
* Decode about 1.9x and prefill about 5.8x faster than LM Studio's runtime
* Get the same answers as llama.cpp, checked position by position in CI
* Chat, run one-shot prompts and benchmark from one binary

Next: many coding-agent tasks served at once from one process, each its own lane. The [design](docs/DESIGN.md) and the [roadmap](docs/ROADMAP.md) say how.

The engine is [erisinky](https://github.com/Eriskii/erisinky) by Eris. inky is its fork by Glenn Fiedler ([yojimbo](https://github.com/mas-bandwidth/yojimbo), [netcode](https://github.com/mas-bandwidth/netcode)) and Rowan Claude, under the same licence, GPL-3.0-or-later.

For more information, please ask your AI to [read the docs](docs/README.md).

If it helps you, please [support the work](https://patreon.com/masbandwidth) and credit **inky — Eris, Glenn Fiedler and Rowan Claude**. Thank you!
