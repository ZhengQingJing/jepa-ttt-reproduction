import time
import numpy as np
import torch
from common import ROOT, TwoRoomEnv, ExpertPolicy


def main():
    torch.set_num_threads(1)
    env = TwoRoomEnv()
    policy = ExpertPolicy(action_noise=2.0, action_repeat_prob=0.05, seed=170041)
    policy.set_env(env)
    poses, commands, offsets = [], [], [0]
    start = time.monotonic()
    for episode in range(3000):
        _, info = env.reset(seed=100000 + episode)
        for t in range(100):
            pos = info["state"].copy()
            action = policy.get_action(info)
            poses.append(pos)
            commands.append(action.copy())
            _, _, done, trunc, info = env.step(action)
            if done or trunc:
                break
        offsets.append(len(poses))
        if (episode + 1) % 250 == 0:
            print(
                f"{episode + 1} episodes, {len(poses)} rows, {time.monotonic() - start:.1f}s",
                flush=True,
            )
    (ROOT / "data").mkdir(exist_ok=True)
    np.savez_compressed(
        ROOT / "data/offline.npz",
        pos=np.array(poses, np.float32),
        action=np.array(commands, np.float32),
        offsets=np.array(offsets, np.int64),
    )
    print("collected", len(poses), "native observation/action rows")


if __name__ == "__main__":
    main()
