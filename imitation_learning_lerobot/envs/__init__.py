from .env import Env
from .env_factory import EnvFactor

from .pick_and_place_env import PickAndPlaceEnv
from .dishwasher_env import DishwasherEnv

EnvFactor.register_all()
