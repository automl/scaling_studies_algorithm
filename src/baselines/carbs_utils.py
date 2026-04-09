import math
import sys

import numpy as np
from loguru import logger
import neps
import copy
from neps.optimizers.optimizer import AskFunction, SampledConfig
from neps.optimizers.algorithms import custom
from neps.state import Trial
import numpy as np
import neps
import copy
from neps.optimizers.optimizer import AskFunction, SampledConfig
from neps.optimizers.algorithms import custom
from neps.state import Trial

    
from carbs import CARBS, CARBSParams, ObservationInParam, Param
    
from carbs import CARBS, CARBSParams, Param
from typing import TYPE_CHECKING, Mapping

from neps.state.optimizer import BudgetInfo
from carbs import CARBS, CARBSParams, LogSpace, Param, LinearSpace, LogitSpace
import neps
logger.remove()
logger.add(sys.stdout, level="DEBUG", format="{message}")

import numpy as np
import copy
from neps.optimizers.optimizer import AskFunction, SampledConfig
from neps.optimizers.algorithms import custom
from neps.state import EvaluatePipelineReturn, Trial, UserResult

    
from carbs import CARBS, CARBSParams, ObservationInParam, Param
import logging
from typing import TYPE_CHECKING, Mapping

from neps.state.optimizer import BudgetInfo

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def convert_space_to_carbs_space(space: neps.PipelineSpace):
    for attr_name, attr_value in space.get_attrs().items():
        if isinstance(attr_value, (neps.Float, neps.Integer)):
            search_center = (attr_value.lower + attr_value.upper) / 2 if not attr_value.has_prior else attr_value.prior
            # search_center = None
            # Check if bounds are within [0, 1] range
            
            if attr_value.log:
                yield Param(
                    name=attr_name,
                    space=LogSpace(
                        scale=- 1/ math.log(attr_value.upper / attr_value.lower),
                        min=attr_value.lower,
                        max=attr_value.upper,
                        is_integer=isinstance(attr_value, neps.Integer),
                    ),
                    search_center=search_center,
                )
            elif 0 <= attr_value.lower and attr_value.upper <= 1:
                yield Param(
                    name=attr_name,
                    space=LogitSpace(
                        scale=1.0,
                        min=attr_value.lower,
                        max=attr_value.upper,
                        is_integer=isinstance(attr_value, neps.Integer),
                    ),
                    search_center=search_center,
                )
            else:
                yield Param(
                    name=attr_name,
                    space=LinearSpace(
                        scale=3 * (attr_value.upper - attr_value.lower),
                        min=attr_value.lower,
                        max=attr_value.upper,
                        is_integer=isinstance(attr_value, neps.Integer),
                    ),
                    search_center=search_center,
                )
        elif isinstance(attr_value, (int, float, str)):
            continue
        else:
            raise NotImplementedError(f"Space type {type(attr_value)} is not supported.")

def evaluate_pipeline_wrapper(carbs_opt, evaluate_pipeline):
    def eval_pipeline(**config):
        res = evaluate_pipeline(**config)
        obs_out = carbs_opt.observe(ObservationInParam(input=config, output=res["objective_to_minimize"], cost=res["cost"]))
        return res
    return eval_pipeline

class CARBSSuggester(AskFunction):
    
    space: neps.PipelineSpace
    
    def __init__(self, space, *, carbs_opt):
        self.carbs = carbs_opt

    def __call__(
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo | None,
        n = None,
    ) -> SampledConfig | list[SampledConfig]:
        suggestion = self.carbs.suggest().suggestion
        sug = copy.deepcopy(suggestion)
        sug.pop("suggestion_uuid")
        return SampledConfig(id=str(len(trials)), config=sug)
    
    def get_trial_artifacts(
        self,
        trials):
        return None

    def import_trials(
        self,
        external_evaluations,
        trials):
        return []
    

def get_carbs_optimizer(space: neps.PipelineSpace, seed):
    param_spaces = list(convert_space_to_carbs_space(space))
    carbs_params = CARBSParams(
        num_random_samples=4,
        better_direction_sign=-1,
        is_wandb_logging_enabled=False,
        resample_frequency=0,
        seed=seed,
        initial_search_radius=0.15,
    )
    carbs_opt = CARBS(carbs_params, param_spaces)
    opt = custom(name="carbs", optimizer=lambda space: CARBSSuggester(space, carbs_opt=carbs_opt))
    return opt, carbs_opt
