# first approach of chinchilla
# have fix parameters list with different training horizons.
# create a custom optimizer for evalauting them in the neps format.
import neps

from benchmarks.synthetic_transformer_space import get_total_flops, get_number_of_parameters, seen_datapoints_estimator, optimizers, evaluate_pipeline, pipeline_space, MAX_TAGET_FLOPS

neps.run(
    evaluate_pipeline=evaluate_pipeline,
    pipeline_space=pipeline_space,
    # root_directory=f"results_test/chinchilla_2",
    root_directory=f"results_test/gemstones",
    cost_to_spend=MAX_TAGET_FLOPS * 3,
    overwrite_root_directory=True,
    optimizer=optimizers["grid_search"],
)
