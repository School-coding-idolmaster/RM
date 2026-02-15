from train.envs.flat_env import FlatEnv
from train.envs.ridged_env import RidgedEnv
from train.envs.beams_env import BeamsEnv
from train.envs.frozen_lake_env import FrozenLakeEnv

__all__ = ["create_env", "FlatEnv", "RidgedEnv", "BeamsEnv", "FrozenLakeEnv"]


def create_env(env, xml, **kwargs):
    """Create a terrain environment for the given robot XML.

    Args:
        env: Terrain type — one of "flat", "ridged", "beams", "frozen_lake".
        xml: Path to a MuJoCo robot XML file.

    Returns:
        A PipelineEnv subclass instance.
    """
    if env == "flat":
        return FlatEnv(xml, **kwargs)
    elif env == "ridged":
        return RidgedEnv(xml, **kwargs)
    elif env == "beams":
        return BeamsEnv(xml, **kwargs)
    elif env == "frozen_lake":
        return FrozenLakeEnv(xml, **kwargs)
    else:
        raise ValueError(f"Unknown environment: {env}")
