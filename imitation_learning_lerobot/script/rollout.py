import logging
import time
from contextlib import nullcontext
from dataclasses import asdict
from pprint import pformat

import re
import torch

from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.common.utils.utils import get_safe_torch_device
from lerobot.common.envs.utils import preprocess_observation

from configs import PickAndPlaceEnvConfig
from envs import PickAndPlaceEnv


@parser.wrap()
def main(cfg: EvalPipelineConfig):
    logging.info(pformat(asdict(cfg)))

    device = get_safe_torch_device(cfg.policy.device, log=True)

    policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env)
    policy.eval()

    env = PickAndPlaceEnv("human")
    observation, info = env.reset()

    while True:
        observation = preprocess_observation(observation)
        observation = {
            key: observation[key].to(device, non_blocking=device.type == "cuda") for key in observation
        }

        with torch.inference_mode():
            action = policy.select_action(observation)

        action = action.to("cpu").numpy()
        action = action.flatten()

        observation, _, _, _, info = env.step(action)
        env.render()
        time.sleep(0.04)


if __name__ == '__main__':
    main()
