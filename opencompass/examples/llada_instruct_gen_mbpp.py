from mmengine.config import read_base
# from opencompass.configs.datasets.mbpp.internal_mbpp_gen_ce6b06 import \
#     internal_mbpp_gen
with read_base():
    from opencompass.configs.datasets.mbpp.mbpp_gen import \
        mbpp_datasets
    from opencompass.configs.models.dllm.llada_instruct_8b import \
        models as llada_instruct_8b_models
datasets = mbpp_datasets
models = llada_instruct_8b_models
eval_cfg = {'gen_blocksize': 32, 'gen_length': 512, 'gen_steps': 512, 'batch_size':1, 'batch_size_':1}
for model in models:
    model.update(eval_cfg)
from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask
infer = dict(
    partitioner=dict(
        type=NumWorkerPartitioner,
        num_worker=8,    # 划分完成后的任务数 / 预期能有的 worker 数
        num_split=None,   # 每个数据集将被划分成多少份。若为 None，则使用 num_worker。
        min_task_size=16, # 每个划分的最小数据条目数
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=64,
        task=dict(type=OpenICLInferTask),
        retry=5
    ),
)
# python run.py examples/llada_gen_bbh.py -w outputs/demo