from mmengine.config import read_base
# /home/xushaoxuan/d-llm/opencompass/opencompass/configs/datasets/mmlu/mmlu_gen_a484b3.py
# answer = 3, block = 3, fewshot = 5
with read_base():
    from opencompass.configs.datasets.mmlu.mmlu_gen_a484b3 import \
        mmlu_datasets
    from opencompass.configs.models.dllm.llada_instruct_8b import \
        models as llada_instruct_8b_models
    from opencompass.configs.summarizers.groups.mmlu import \
        mmlu_summary_groups
datasets = mmlu_datasets
models = llada_instruct_8b_models
summarizer = dict(
    summary_groups=sum([v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)
eval_cfg = {'gen_blocksize': 3, 'gen_length': 3, 'gen_steps': 3, 'batch_size_': 2, 'batch_size': 2}
for model in models:
    model.update(eval_cfg)
from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask
infer = dict(
    partitioner=dict(
        type=NumWorkerPartitioner,
        num_worker=6,    # 划分完成后的任务数 / 预期能有的 worker 数
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