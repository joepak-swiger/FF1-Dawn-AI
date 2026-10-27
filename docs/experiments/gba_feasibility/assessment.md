# Dawn of Souls GBA feasibility assessment

**Status: exploratory / no-go for training.** This is an independent track from
the NES FF1 work. GBA changes live on `codex/gba-feasibility` until reviewed.

## Evidence observed locally

- The custom Stable-Retro integration is registered and historical scripts
  report that it booted, accepted inputs, and could serialize/restore a state
  created inside its own emulator core. Those old checks did not prove a
  playable post-party checkpoint or a deterministic restart/resume.
- Two local mGBA `.ss0` files are PNG containers. Their chunk inventories
  include `gbAs` (92,809 and 68,585 payload bytes) and `gbAx`; the separate
  Stable-Retro `.state` example is 397,312 bytes and does not have a PNG
  signature. A read-only inspector is available as
  `scripts/gba_feasibility_probe.py`.
- Historical screenshots after importing the external `.ss0` files show the
  Square Enix logo screen. This confirms that those imports did not reach the
  expected post-party scene. It does not by itself explain why: core version,
  serialization format, game state, timing, or state-file conversion may each
  contribute.
- The existing battery probe copied a 32 KiB battery-save file into several
  guessed filenames in the live custom integration. That experiment is
  inconclusive and modified integration contents; the new probe never writes
  there.

## Fresh test approach

Run the following in the existing WSL environment when it is available:

```bash
source .venv/bin/activate
python scripts/gba_feasibility_probe.py --runtime --automate-party-setup
```

The probe opens the registered GBA integration, advances a fixed START/A setup
sequence, optionally replays the previously explored name-entry macro, and
serializes the state directly from the running Stable-Retro core. It then
resets that core, restores the newly generated state, repeats an identical
input sequence, and requires exact frame equality. Screenshots, state, JSON,
and text report are stored in a new `videos/gba_feasibility/<run-id>/` folder,
which is ignored by Git. The saved screenshot is a **candidate** checkpoint;
the macro is not proof that party creation succeeded. Confirm the screen
visually before treating it as playable.

The format-only path works without Stable-Retro:

```bash
python scripts/gba_feasibility_probe.py
```

It reports available local artifacts without dumping their contents. Missing
files are reported as missing. Neither mode edits the ROM, integration files,
existing saves, or prior experiment outputs.

## Battery-save assessment

Upstream [Stable-Retro emulator source](https://github.com/Farama-Foundation/stable-retro/blob/main/src/emulator.cpp)
currently handles `RETRO_MEMORY_SYSTEM_RAM` and native core serialization;
the file contains no `RETRO_ENVIRONMENT_GET_SAVE_DIRECTORY` or
`RETRO_MEMORY_SAVE_RAM` handling. Upstream [mGBA core documentation](https://docs.libretro.com/library/mgba/)
describes battery saves as frontend save-directory `.srm` files. Together,
these sources suggest Stable-Retro may not persist libretro battery SRAM in
the expected frontend manner. This is a hypothesis about the current local
runtime until verified against the exact installed Stable-Retro build and a
controlled isolated test.

Do not retry guessed save filenames in the working integration. If native
battery persistence is still needed, the next test should clone the integration
into the unique output folder, place one candidate save only in that copy, and
observe whether the core exposes or writes save RAM across reset. If the
Stable-Retro API offers no path to the libretro save-memory interface, prioritize
the core-native serialized checkpoint for RL episode starts and treat battery
saves as unsupported by this frontend.

## Decision gate

GBA becomes a viable RL foundation only after all of these checks pass:

1. Repeatable boot and reproducible control mapping.
2. A screenshot-confirmed playable state after party creation.
3. A state created by the active Stable-Retro core that survives reset,
   reload, and deterministic resume.
4. A clear battery-save decision: demonstrated persistence, a safe workaround,
   or confirmation that battery persistence is unnecessary for episode starts.
5. Stable-enough frames per second and reset behavior for repeated RL rollouts.

Until then, preserve the working NES path and do not start large-scale GBA
training. See GitHub [Issue #1](https://github.com/joepak-swiger/FF1-Dawn-AI/issues/1)
for the requested spike scope.
