from typing import Type
from pathlib import Path
import dataclasses
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from imitation_learning_lerobot.envs import Env, EnvFactor


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


def create_empty_dataset(env_cls: Type[Env]):
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


def populate_dataset(env_cls: Type[Env], dataset: LeRobotDataset):
    episode = 100

    env = env_cls()
    task = env.name
    for i in range(episode):
        data = env.run()

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

    env.close()


def main():
    name = "dishwasher"
    env_cls = EnvFactor.get_strategies(name)

    dataset = create_empty_dataset(env_cls)

    populate_dataset(env_cls, dataset)


if __name__ == '__main__':
    main()
