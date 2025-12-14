from enum import Enum


class Env(str, Enum):
    STARCRAFT = "starcraft"
    PETTINGZOO = "pettingzoo"
    GRF = "football"
    MAMUJOCO = "mamujoco"
    SMAX = "smax"

RANDOM_SEED = 23
