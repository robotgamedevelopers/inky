[![ci](https://github.com/robotgamedevelopers/inky/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/robotgamedevelopers/inky/actions/workflows/ci.yml)

<img width="2172" height="724" alt="image" src="https://github.com/user-attachments/assets/1ae045b5-9adc-4590-8746-bd55b8bedad8" />

Fast local inference for [**Inkling 2 Mini**](https://thinkingmachines.ai/news/inkling-2-mini/) on Apple silicon.

With **inky**, you can:

* Run Thinking Machines' Inkling 2 Mini on an M5 Mac
* Match llama.cpp argmaxes (worst KL <= 0.01) across CI's checked reference positions.
* Serve many coding-agent tasks at once over HTTP from one process.

[![Matched 20K-context throughput vs. parallel lanes on Apple M5 Max (128 GB)](docs/assets/readme-lanes-speed.svg)](docs/PERFORMANCE.md)

For more details see [performance](docs/PERFORMANCE.md), [design](docs/DESIGN.md), and [roadmap](docs/ROADMAP.md).

Built by Glenn Fiedler ([yojimbo](https://github.com/mas-bandwidth/yojimbo), [netcode](https://github.com/mas-bandwidth/netcode)), Rowan Claude and [Eris](https://github.com/Eriskii/erisinky).

For more information, please ask your AI to [read the docs](docs/README.md), or see [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for building and running the validation suite with or without an M5 Mac.
