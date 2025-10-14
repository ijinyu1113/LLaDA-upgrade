from opencompass.models import LLaDABaseModel

models = [
    dict(
        type=LLaDABaseModel,
        abbr='llada-8b-base',
        path='/path/to/LLaDA-8B-Base',
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    )
]
