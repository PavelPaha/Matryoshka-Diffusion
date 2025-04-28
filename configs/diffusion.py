from dataclasses import dataclass, field
from enum import Enum


class ScheduleType(Enum):
    DEEPFLOYD = 0
    DDPM = 1


class PredictionType(Enum):
    DDPM = 0
    V_PREDICTION = 1


@dataclass
class DiffusionConfig:
    num_diffusion_steps: int = 1000
    schedule_type: ScheduleType = ScheduleType.DEEPFLOYD
    prediction_type: PredictionType = PredictionType.V_PREDICTION
    beta_start: float = 0.0001
    beta_end: float = 0.02
    use_double_loss: bool = True
    multi_res_weights: list[float] = field(default_factory=lambda: [4.0, 1.0])
