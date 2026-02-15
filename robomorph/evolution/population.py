import pydantic
from typing import List, Optional


class EvolutionUnit(pydantic.BaseModel):
    """An individual robot design in the population."""
    prompt: str = ""                         # task description for LLM
    structure: Optional[str] = None          # structural graph (e.g., "H-U-(C-M-E)-Y-U-T")
    components: Optional[str] = None         # component graph (e.g., "[null]-[body=15cm]-...")
    fitness: float = -1.0                    # RL-evaluated locomotion fitness (-1 = not evaluated)
    diversity: Optional[int] = None          # design diversity score
    xml_path: Optional[str] = None           # path to compiled MuJoCo XML


class Population(pydantic.BaseModel):
    """A population of robot designs with evolutionary state."""
    size: int
    problem_description: str
    elites: List[EvolutionUnit]
    units: List[EvolutionUnit]
