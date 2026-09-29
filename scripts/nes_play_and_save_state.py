from pathlib import Path

import numpy as np
from PIL import Image
import pygame
import stable_retro

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CUSTOM_INTEGRATIONS = PROJECT_ROOT / "custom_integrations"
GAME = "FinalFantasy-Nes"

OUT_DIR = PROJECT_ROOT / "videos" / "nes_play_and_save_state"
STATE_DIR = PROJECT_ROOT / "states"
STATE_PATH = STATE_DIR / "ff1_nes_cornelia_start.state"

OUT_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR.mkdir(parents=True, exist_ok=True)

# NES mapping from our boot test:
# ['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']
KEY_TO_INDEX = {
    pygame.K_z: 8,             # A
    pygame.K_x: 0,             # B

    pygame.K_RETURN: 3,        # START
    pygame.K_KP_ENTER: 3,
    pygame.K_SPACE: 3,
    pygame.K_p: 3,

    pygame.K_BACKSPACE: 2,     # SELECT
    pygame.K_TAB: 2,
    pygame.K_LSHIFT: 2,
    pygame.K_RSHIFT: 2,

    pygame.K_UP: 4,
    pygame.K_DOWN: 5,
    pygame.K_LEFT: 6,
    pygame.K_RIGHT: 7,
}

SCALE = 3
FPS = 60

stable_retro.data.Integrations.add_custom_path(str(CUSTOM_INTEGRATIONS))

env = stable_retro.make(
    game=GAME,
    state=stable_retro.State.NONE,
    inttype=stable_retro.data.Integrations.ALL,
    render_mode="rgb_array",
)

reset_result = env.reset()
obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result

pygame.init()
height, width = obs.shape[:2]
screen = pygame.display.set_mode((width * SCALE, height * SCALE))
pygame.display.set_caption("FF1 NES AI - Play and Save State")

clock = pygame.time.Clock()
pressed_keys = set()


def make_action():
    action = np.zeros(env.action_space.shape, dtype=np.int8)

    for key in pressed_keys:
        index = KEY_TO_INDEX.get(key)
        if index is not None:
            action[index] = 1

    return action


def step_env(action):
    global obs

    result = env.step(action)

    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        done = terminated or truncated
    else:
        obs, reward, done, info = result

    return done


def render():
    surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
    surface = pygame.transform.scale(surface, (width * SCALE, height * SCALE))
    screen.blit(surface, (0, 0))
    pygame.display.flip()


def save_screen(name):
    path = OUT_DIR / name
    Image.fromarray(obs).save(path)
    Image.fromarray(obs).save(OUT_DIR / "current.png")
    print("Saved screen:", path)


def save_state():
    STATE_PATH.write_bytes(env.em.get_state())
    save_screen("saved_state_screen.png")
    print()
    print("SAVED NES STABLE-RETRO STATE:")
    print(STATE_PATH)
    print("State size:", STATE_PATH.stat().st_size, "bytes")
    print()


print()
print("FF1 NES - Live State Builder")
print("----------------------------")
print("Controls:")
print("  Z = A")
print("  X = B")
print("  Enter / Space / P = START")
print("  Shift / Backspace / Tab = SELECT")
print("  Arrow keys = D-pad")
print("  F5 = save state")
print("  F9 = screenshot")
print("  Esc = quit")
print()
print("Goal: play until you are outside Cornelia / on the overworld, then press F5.")
print()

save_screen("000_start.png")
render()

running = True
frame = 0

try:
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                print("DOWN:", pygame.key.name(event.key), "code:", event.key)

                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_F5:
                    save_state()
                elif event.key == pygame.K_F9:
                    save_screen(f"screenshot_{frame:06d}.png")
                else:
                    pressed_keys.add(event.key)

            elif event.type == pygame.KEYUP:
                print("UP:", pygame.key.name(event.key), "code:", event.key)
                pressed_keys.discard(event.key)

        done = step_env(make_action())

        if done:
            print("Environment ended. Resetting.")
            reset_result = env.reset()
            obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result

        render()
        frame += 1
        clock.tick(FPS)

finally:
    env.close()
    pygame.quit()
    print("Closed emulator.")
