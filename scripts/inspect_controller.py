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

print("Action space:", env.action_space)
print("Buttons:")

buttons = getattr(env, "buttons", None)

if buttons is None:
    print("Could not find env.buttons")
else:
    for i, button in enumerate(buttons):
        print(f"{i}: {button}")

env.close()
