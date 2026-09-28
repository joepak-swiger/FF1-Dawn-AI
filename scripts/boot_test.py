from pathlib import Path
import stable_retro

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CUSTOM_INTEGRATIONS = PROJECT_ROOT / "custom_integrations"
GAME = "FinalFantasyDawnOfSouls-GbAdvance"

stable_retro.data.Integrations.add_custom_path(str(CUSTOM_INTEGRATIONS))

games = stable_retro.data.list_games(inttype=stable_retro.data.Integrations.ALL)

print("Custom integrations path:", CUSTOM_INTEGRATIONS)
print("Game registered:", GAME in games)

if GAME not in games:
    print("Final Fantasy-ish games found:")
    for game in games:
        if "Final" in game or "Fantasy" in game or "Dawn" in game:
            print(" -", game)
    raise SystemExit("Game was not registered.")

env = stable_retro.make(
    game=GAME,
    state=stable_retro.State.NONE,
    inttype=stable_retro.data.Integrations.ALL,
    render_mode="rgb_array",
)

reset_result = env.reset()
if isinstance(reset_result, tuple):
    obs = reset_result[0]
else:
    obs = reset_result

print("Environment created:", env)
print("Action space:", env.action_space)
print("Observation type:", type(obs))
print("Observation shape:", getattr(obs, "shape", None))

for i in range(30):
    action = env.action_space.sample()
    step_result = env.step(action)

    if len(step_result) == 5:
        obs, reward, terminated, truncated, info = step_result
        done = terminated or truncated
    else:
        obs, reward, done, info = step_result

    if done:
        print("Episode ended early at step", i)
        break

print("Boot test finished successfully.")
env.close()
