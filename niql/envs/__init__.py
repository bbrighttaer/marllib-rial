from marllib.envs.base_env import ENV_REGISTRY
from marllib.envs.global_reward_env import COOP_ENV_REGISTRY

from .switch_game import SwitchGame
from .utils import make_local_env
from ..config import SWITCH_GAME


def get_active_env(**kwargs):
    return make_switch_game_env(**kwargs)


def make_switch_game_env(**kwargs):
    # register new env
    ENV_REGISTRY["SwitchGame"] = SwitchGame
    COOP_ENV_REGISTRY["SwitchGame"] = SwitchGame

    # choose environment + scenario
    env = make_local_env(
        environment_name="SwitchGame",
        map_name="all_scenario",
        **kwargs,
    )
    return env, SWITCH_GAME
