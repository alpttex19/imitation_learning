import dataclasses
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from imitation_learning_lerobot.envs import PickAndPlaceEnv


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


def create_empty_dataset():
    features = {"observation.state": {
        "dtype": "float32",
        "shape": (4,),
        "names": {
            "position": ["px",
                         "py",
                         "pz",
                         "grasp"],
        }
    }, "action": {
        "dtype": "float32",
        "shape": (4,),
        "names": {
            "position": ["px",
                         "py",
                         "pz",
                         "grasp"],
        }
    }, "observation.images.top": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": [
            "height",
            "width",
            "channel"
        ]
    }, "observation.images.hand": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": [
            "height",
            "width",
            "channel"
        ]
    }
    }

    config = DatasetConfig()

    dataset = LeRobotDataset.create(
        repo_id="pick_and_place",
        fps=25,
        features=features,
        root="../../outputs/pick_and_place",
        robot_type="UR5e",
        use_videos=config.use_videos,
        tolerance_s=config.tolerance_s,
        image_writer_processes=config.image_writer_processes,
        image_writer_threads=config.image_writer_threads,
        video_backend=config.video_backend
    )

    return dataset


def populate_dataset(dataset: LeRobotDataset):
    episode = 1

    env = PickAndPlaceEnv()
    task = "pick_and_place"
    for i in range(episode):
        data = env.run()

        episode_length = len(data["observations"])

        for j in range(episode_length):
            frame = {
                "observation.state": data["observations"][j]["agent_pos"],
                "action": data["actions"][j],
                "observation.images.top": data["observations"][j]["pixels"]["top"],
                "observation.images.hand": data["observations"][j]["pixels"]["hand"]
            }
            dataset.add_frame(frame, task=task)
        dataset.save_episode()

    env.close()


def main():
    dataset = create_empty_dataset()

    populate_dataset(dataset)


if __name__ == '__main__':
    main()
