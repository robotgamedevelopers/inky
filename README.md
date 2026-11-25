[![ci](https://github.com/robotgamedevelopers/inky/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/robotgamedevelopers/inky/actions/workflows/ci.yml)

<img width="2172" height="724" alt="image" src="https://github.com/user-attachments/assets/1ae045b5-9adc-4590-8746-bd55b8bedad8" />

Fast local inference for [**Inkling 2 Mini**](https://thinkingmachines.ai/news/inkling-2-mini/) on Apple silicon.

With **inky**, you can:

* Run Thinking Machines' Inkling 2 Mini on an M5 Mac, from the LM Studio GGUF
* Decode about 1.9x and prefill about 5.8x faster than LM Studio's runtime
* Match llama.cpp argmaxes (worst KL <= 0.01) across CI's checked reference positions: 58/58 on short-prompt decode, 24/24 after a 2380-token long-prompt prefill, and 58/58 on short-prompt decode steps
* Run many coding-agent tasks served at once from one process, each its own lane. The [design](docs/DESIGN.md) and the [roadmap](docs/ROADMAP.md) say how.

This is a fork of [erisinky](https://github.com/Eriskii/erisinky) by Eris.

Built by Glenn Fiedler ([yojimbo](https://github.com/mas-bandwidth/yojimbo), [netcode](https://github.com/mas-bandwidth/netcode)), Rowan Claude and Eris.

For more information, please ask your AI to [read the docs](docs/README.md), or see [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for building and running the validation suite with or without an M5 Mac.

## Licensing (`GPL-3.0-or-later`)

`inky` (and the `inky-engine` crate) is free software distributed under the [GNU General Public License v3.0 or later](LICENSE), inherited from upstream `erisinky`. Linking, embedding, or redistributing `inky` or `inky-engine` in a binary or library requires distributing the combined work's complete corresponding source code under `GPL-3.0-or-later`-compatible terms. Calling `inky serve` as a separate process over its HTTP APIs (`/v1/chat/completions`, `/v1/messages`, `/health`) sends prompts and receives generated tokens over standard network protocols without linking `inky` into the client program, though any redistributed copy or modified version of `inky` itself remains covered by `GPL-3.0-or-later`.
