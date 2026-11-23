[![ci](https://github.com/robotgamedevelopers/inky/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/robotgamedevelopers/inky/actions/workflows/ci.yml)

<img width="2172" height="724" alt="image" src="https://github.com/user-attachments/assets/1ae045b5-9adc-4590-8746-bd55b8bedad8" />

Fast local inference for **Inkling 2 Mini** on Apple silicon.

With **inky**, you can:

* Run Thinking Machines' Inkling 2 Mini on an M5 Mac, from the LM Studio GGUF
* Decode about 1.9x and prefill about 5.8x faster than LM Studio's runtime
* Get the same answers as llama.cpp, checked position by position in CI
* Run many coding-agent tasks served at once from one process, each its own lane. The [design](docs/DESIGN.md) and the [roadmap](docs/ROADMAP.md) say how.

This is a fork of [erisinkling](https://github.com/Eriskii/erisinkling) by Eris.

Built by Glenn Fiedler ([yojimbo](https://github.com/mas-bandwidth/yojimbo), [netcode](https://github.com/mas-bandwidth/netcode)), Rowan Claude and Eris.

For more information, please ask your AI to [read the docs](docs/README.md).
