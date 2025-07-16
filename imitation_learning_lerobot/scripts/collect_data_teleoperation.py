import time
from typing import Type
from pathlib import Path
import argparse
import dataclasses

import numpy as np
import cv2

from imitation_learning_lerobot.envs import Env, EnvFactor, PickBoxEnv
from imitation_learning_lerobot.utils.key_listener2 import KeyListener
from imitation_learning_lerobot.utils.real_time_sync import RealTimeSync


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


class MotionKeyListener(KeyListener):
    def __init__(self):
        super().__init__()
        self._action0 = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        self._sync = False
        self._done = False
        self._vel = 0.005

    def on_press(self, key):
        super().on_press(key)

        for key_char in self.active_keys:

            try:
                if key_char == 'Key.shift_r':
                    self._sync = not self._sync

                if key_char == 'Key.space':
                    self._done = True

                if not self._sync:
                    break
                if key_char.lower() == '2':
                    self._action[2] += self._vel

                if key_char.lower() == '8':
                    self._action[2] -= self._vel

                if key_char.lower() == "6":
                    self._action[0] -= self._vel

                if key_char.lower() == '4':
                    self._action[0] += self._vel

                if key_char.lower() == '7':
                    self._action[1] += self._vel

                if key_char.lower() == '1':
                    self._action[1] -= self._vel

                if key_char.lower() == '9':
                    self._action[3] += 0.05

                if key_char.lower() == '3':
                    self._action[3] -= 0.05
            except AttributeError:
                pass

    def reset(self):
        self._action[:] = self._action0
        self._sync = False
        self._done = False

    @property
    def action(self):
        return self._action.copy()

    @property
    def sync(self):
        return self._sync

    @property
    def done(self):
        return self._done


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--env.type',
        type=str,
        dest='env_type',
        required=True,
        help='env type'
    )

    parser.add_argument(
        '--episode',
        type=int,
        default=100,
        help='episode'
    )

    return parser.parse_args()


def create_empty_dataset(env_cls: Type[Env]):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(env_cls.states),),
            "names": {
                "position": env_cls.states,
            }
        }, "action": {
            "dtype": "float32",
            "shape": (len(env_cls.states),),
            "names": {
                "position": env_cls.states,
            }
        }
    }

    for camera in env_cls.cameras:
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": (env_cls.height, env_cls.width, 3),
            "names": [
                "height",
                "width",
                "channel"
            ]
        }

    config = DatasetConfig()

    dataset = LeRobotDataset.create(
        repo_id=env_cls.name,
        fps=env_cls.control_hz,
        features=features,
        root=Path(__file__).parent.parent.parent / Path("outputs/datasets") / Path(env_cls.name),
        robot_type=env_cls.robot_type,
        use_videos=config.use_videos,
        tolerance_s=config.tolerance_s,
        image_writer_processes=config.image_writer_processes,
        image_writer_threads=config.image_writer_threads,
        video_backend=config.video_backend
    )

    return dataset


def populate_dataset(episode: int, env_cls: Type[Env], dataset):
    env = env_cls(render_mode="human")
    task = env.name

    rt_sync = RealTimeSync(1.0 / env.control_hz)

    listener = MotionKeyListener()
    listener.start()

    for i in range(episode):

        env.reset()
        listener.reset()

        observations = []
        actions = []

        while not listener.done:

            start = time.time()
            if not listener.sync:
                continue

            action = listener.action
            observation, reward, terminated, truncated, info = env.step(action)
            actions.append(action)
            observations.append(observation)

            env.render()

            for camera in env_cls.cameras:
                cv2.imshow(camera, cv2.cvtColor(observation["pixels"][camera], cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

            print(time.time() - start)
            rt_sync.sync()

        env.close()

        data = {
            "observations": observations,
            "actions": actions
        }

        episode_length = len(data["observations"])

        for j in range(episode_length):
            frame = {
                "observation.state": data["observations"][j]["agent_pos"],
                "action": data["actions"][j],
            }

            for camera in env_cls.cameras:
                frame[f"observation.images.{camera}"] = data["observations"][j]["pixels"][camera]

            dataset.add_frame(frame, task=task)

        dataset.save_episode()

    cv2.destroyAllWindows()
    listener.stop()

def main():
    args = parse_args()

    env_type = args.env_type
    env_cls = EnvFactor.get_strategies(env_type)

    for camera in env_cls.cameras:
        cv2.namedWindow(camera, cv2.WINDOW_GUI_NORMAL)

    dataset = create_empty_dataset(env_cls)

    populate_dataset(args.episode, env_cls, dataset)


if __name__ == '__main__':
    main()
