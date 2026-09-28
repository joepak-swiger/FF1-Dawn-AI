from pathlib import Path
import stable_retro

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CUSTOM_INTEGRATIONS = PROJECT_ROOT / "custom_integrations"
GAME = "FinalFantasyDawnOfSouls-GbAdvance"

stable_retro.data.Integrations.add_custom_path(str(CUSTOM_INTEGRATIONS))

env = stable_retro.make(
    game=GAME,
    state=stable_retro.State.NONE,
    inttype=stable_retro.data.Integrations.ALL,
    render_mode="rgb_array",
)

print("Environment type:", type(env))
print()
print("Environment attributes containing 'state', 'save', or 'load':")
for name in dir(env):
    lowered = name.lower()
    if "state" in lowered or "save" in lowered or "load" in lowered:
        print(" -", name)

print()
print("Emulator object type:", type(env.em))
print()
print("Emulator attributes containing 'state', 'save', or 'load':")
for name in dir(env.em):
    lowered = name.lower()
    if "state" in lowered or "save" in lowered or "load" in lowered:
        print(" -", name)

env.close()
