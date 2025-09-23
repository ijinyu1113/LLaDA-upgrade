from mmengine.config import read_base
# /home/xushaoxuan/d-llm/opencompass/opencompass/configs/datasets/mmlu/mmlu_gen_a484b3.py
# answer = 256, block = 256, fewshot = 0
with read_base():
    from opencompass.configs.datasets.mmlu_pro.mmlu_pro_gen import \
        mmlu_pro_datasets
    from opencompass.configs.models.dllm.llada_instruct_8b import \
        models as llada_instruct_8b_models
    from opencompass.configs.summarizers.groups.mmlu_pro import \
        mmlu_pro_summary_groups
datasets = mmlu_pro_datasets
models = llada_instruct_8b_models
summarizer = dict(
    summary_groups=sum([v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)
eval_cfg = {'gen_blocksize': 256, 'gen_length': 256, 'gen_steps': 256, 'batch_size':1, 'batch_size_':1}
for model in models:
    model.update(eval_cfg)
from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask
infer = dict(
    partitioner=dict(
        type=NumWorkerPartitioner,
        num_worker=7,    # 划分完成后的任务数 / 预期能有的 worker 数
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