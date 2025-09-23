import os
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import transformers

from opencompass.models.base import BaseModel
from opencompass.models.base_api import APITemplateParser
from opencompass.registry import MODELS
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList
import torch.nn.functional as F
import numpy as np
PromptType = Union[PromptList, str]
# @MODELS.register_module()
# class LLaDAModel(BaseModel):
def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens
#     def __init__(self,
#                  pkg_root: str,
#                  ckpt_path: str,
#                  tokenizer_only: bool = False,
#                  meta_template: Optional[Dict] = None,
#                  **kwargs):
#         super().__init__(pkg_root, ckpt_path, tokenizer_only, meta_template, **kwargs)

#     def add_gumbel_noise(logits, temperature):
#         '''
#         The Gumbel max is a method for sampling categorical distributions.
#         According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
#         Thus, we use float64.
#         '''
#         if temperature == 0:
#             return logits
#         logits = logits.to(torch.float64)
#         noise = torch.rand_like(logits, dtype=torch.float64)
#         gumbel_noise = (- torch.log(noise)) ** temperature
#         return logits.exp() / gumbel_noise


#     def get_num_transfer_tokens(mask_index, steps):
#         '''
#         In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
#         Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
#         the expected number of tokens transitioned at each step should be consistent.

#         This function is designed to precompute the number of tokens that need to be transitioned at each step.
#         '''
#         mask_num = mask_index.sum(dim=1, keepdim=True)

#         base = mask_num // steps
#         remainder = mask_num % steps

#         num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

#         for i in range(mask_num.size(0)):
#             num_transfer_tokens[i, :remainder[i]] += 1

#         return num_transfer_tokens


#     @ torch.no_grad()
#     def _generate(self, prompt, steps=1024, gen_length=1024, block_length=1024, temperature=0.,
#                 cfg_scale=0., remasking='low_confidence', mask_id=126336):
#         '''
#         Args:
#             model: Mask predictor.
#             prompt: A tensor of shape (1, L).
#             steps: Sampling steps, less than or equal to gen_length.
#             gen_length: Generated answer length.
#             block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
#             temperature: Categorical distribution sampling temperature.
#             cfg_scale: Unsupervised classifier-free guidance scale.
#             remasking: Remasking strategy. 'low_confidence' or 'random'.
#             mask_id: The toke id of [MASK] is 126336.
#         '''
#         x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
#         x[:, :prompt.shape[1]] = prompt.clone()

#         prompt_index = (x != mask_id)

#         assert gen_length % block_length == 0
#         num_blocks = gen_length // block_length

#         assert steps % num_blocks == 0
#         steps = steps // num_blocks

#         for num_block in range(num_blocks):
#             block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
#             num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
#             for i in range(steps):
#                 mask_index = (x == mask_id)
#                 if cfg_scale > 0.:
#                     un_x = x.clone()
#                     un_x[prompt_index] = mask_id
#                     x_ = torch.cat([x, un_x], dim=0)
#                     logits = self.model(x_).logits
#                     logits, un_logits = torch.chunk(logits, 2, dim=0)
#                     logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
#                 else:
#                     logits = self.model(x).logits

#                 logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
#                 x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

#                 if remasking == 'low_confidence':
#                     p = F.softmax(logits, dim=-1)
#                     x0_p = torch.squeeze(
#                         torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
#                 elif remasking == 'random':
#                     x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
#                 else:
#                     raise NotImplementedError(remasking)

#                 x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

#                 x0 = torch.where(mask_index, x0, x)
#                 confidence = torch.where(mask_index, x0_p, -np.inf)

#                 transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
#                 for j in range(confidence.shape[0]):
#                     _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
#                     transfer_index[j, select_index] = True
#                 x[transfer_index] = x0[transfer_index]

#         return x
#     def get_token_len(self, prompt: str) -> int:
#         """Get lengths of the tokenized strings."""
#         return len(self.tokenizer.encode(prompt))

#     def generate(self, inputs: List[str], max_out_len: int) -> List[str]:
#         """Generate results given a list of inputs. """
#         return self._generate(inputs, max_out_len)

#     def get_ppl(self,
#                 inputs: List[str],
#                 mask_length: Optional[List[int]] = None) -> List[float]:
#         """Get perplexity scores given a list of inputs."""
#         pass

@MODELS.register_module()
class LLaDAModel(BaseModel):
    """Model wrapper around HuggingFace models.

    Args:
        path (str): The name or path to HuggingFace's model.
        hf_cache_dir: Set the cache dir to HF model cache dir. If None, it will
            use the env variable HF_MODEL_HUB. Defaults to None.
        max_seq_len (int): The maximum length of the input sequence. Defaults
            to 2048.
        tokenizer_path (str): The path to the tokenizer. Defaults to None.
        tokenizer_kwargs (dict): Keyword arguments for the tokenizer.
            Defaults to {}.
        peft_path (str, optional): The name or path to the HuggingFace's PEFT
            model. If None, the original model will not be converted to PEFT.
            Defaults to None.
        tokenizer_only (bool): If True, only the tokenizer will be initialized.
            Defaults to False.
        model_kwargs (dict): Keyword arguments for the model, used in loader.
            Defaults to dict(device_map='auto').
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        extract_pred_after_decode (bool): Whether to extract the prediction
            string from the decoded output string, instead of extract the
            prediction tokens before decoding. Defaults to False.
        batch_padding (bool): If False, inference with be performed in for-loop
            without batch padding.
        pad_token_id (int): The id of the padding token. Defaults to None. Use
            (#vocab + pad_token_id) if get negative value.
        mode (str, optional): The method of input truncation when input length
            exceeds max_seq_len. 'mid' represents the part of input to
            truncate. Defaults to 'none'.
        use_fastchat_template (str, optional): Whether to use fastchat to get
            the conversation template. If True, fastchat needs to be
            implemented first. Defaults to False.
        end_str (str, optional): Whether to trim generated strings with end_str
            if the model has special ending strings that are not handled well.
            Defaults to None.

    Note:
        About ``extract_pred_after_decode``: Commonly, we should extract the
        the prediction tokens before decoding. But for some tokenizers using
        ``sentencepiece``, like LLaMA,  this behavior may change the number of
        whitespaces, which is harmful for Python programming tasks.
    """

    def __init__(self,
                 path: str,
                 hf_cache_dir: Optional[str] = None,
                 max_seq_len: int = 2048,
                 tokenizer_path: Optional[str] = None,
                 tokenizer_kwargs: dict = dict(),
                 peft_path: Optional[str] = None,
                 tokenizer_only: bool = False,
                 model_kwargs: dict = dict(device_map='auto'),
                 generation_kwargs: dict = dict(),
                 meta_template: Optional[Dict] = None,
                 extract_pred_after_decode: bool = False,
                 batch_padding: bool = False,
                 pad_token_id: Optional[int] = None,
                 mode: str = 'none',
                 use_fastchat_template: bool = False,
                 end_str: Optional[str] = None,
                 stop_words: Optional[List[str]] = [],
                 cfg = 0,
                 temperature = 0.,
                 remasking = 'low_confidence', # 'random'
                 mask_id = 126336, # The token id of [MASK] is 126336.
                 padding_id = 126081, # The token id of <pad> is 126081.
                 mc_num = 1,
                 gen_steps = 512,
                 gen_length = 512,
                 gen_blocksize = 512,
                 batch_size_ = 1,
                 ) -> None:
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         tokenizer_only=tokenizer_only,
                         meta_template=meta_template)
        if hf_cache_dir is None:
            hf_cache_dir = os.getenv('HF_MODEL_HUB', None)
        self.logger = get_logger()
        self.pad_token_id = pad_token_id
        assert mode in ['none', 'mid']
        self.mode = mode
        self._load_tokenizer(path=path,
                             tokenizer_path=tokenizer_path,
                             tokenizer_kwargs=tokenizer_kwargs)
        self.batch_padding = batch_padding
        self.extract_pred_after_decode = extract_pred_after_decode
        if not tokenizer_only:
            self._load_model(path=path,
                             model_kwargs=model_kwargs,
                             peft_path=peft_path)
        self.generation_kwargs = generation_kwargs
        self.use_fastchat_template = use_fastchat_template
        self.end_str = end_str
        self.stop_words = stop_words
        ## add cfg and mc_num
        self.cfg = cfg
        self.mc_num = mc_num

        ##Todo: modify input
        self.batch_size = batch_size_
        self.gen_steps = gen_steps
        self.gen_length = gen_length
        self.gen_blocksize = gen_blocksize
        self.temperature = temperature
        self.remasking = remasking
        self.padding_id = padding_id
        self.mask_id = mask_id

    def _load_tokenizer(self, path: str, tokenizer_path: Optional[str],
                        tokenizer_kwargs: dict):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path if tokenizer_path else path, **tokenizer_kwargs)

        # A patch for some models without pad_token_id
        if self.pad_token_id is not None:
            if self.pad_token_id < 0:
                self.pad_token_id += self.tokenizer.vocab_size
            if self.tokenizer.pad_token_id is None:
                self.logger.debug(f'Using {self.pad_token_id} as pad_token_id')
            elif self.tokenizer.pad_token_id != self.pad_token_id:
                self.logger.warning(
                    'pad_token_id is not consistent with the tokenizer. Using '
                    f'{self.pad_token_id} as pad_token_id')
            self.tokenizer.pad_token_id = self.pad_token_id
        elif self.tokenizer.pad_token_id is None:
            self.logger.warning('pad_token_id is not set for the tokenizer.')
            if self.tokenizer.eos_token is not None:
                self.logger.warning(
                    f'Using eos_token_id {self.tokenizer.eos_token} '
                    'as pad_token_id.')
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                from transformers.generation import GenerationConfig
                gcfg = GenerationConfig.from_pretrained(path)

                if gcfg.pad_token_id is not None:
                    self.logger.warning(
                        f'Using pad_token_id {gcfg.pad_token_id} '
                        'as pad_token_id.')
                    self.tokenizer.pad_token_id = gcfg.pad_token_id
                else:
                    raise ValueError(
                        'pad_token_id is not set for this tokenizer. Try to '
                        'set pad_token_id via passing '
                        '`pad_token_id={PAD_TOKEN_ID}` in model_cfg.')

        # A patch for llama when batch_padding = True
        if 'decapoda-research/llama' in path or \
                (tokenizer_path and
                 'decapoda-research/llama' in tokenizer_path):
            self.logger.warning('We set new pad_token_id for LLaMA model')
            # keep consistent with official LLaMA repo
            # https://github.com/google/sentencepiece/blob/master/python/sentencepiece_python_module_example.ipynb  # noqa
            self.tokenizer.bos_token = '<s>'
            self.tokenizer.eos_token = '</s>'
            self.tokenizer.pad_token_id = 0

    def _set_model_kwargs_torch_dtype(self, model_kwargs):
        if 'torch_dtype' not in model_kwargs:
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = {
                'torch.float16': torch.float16,
                'torch.bfloat16': torch.bfloat16,
                'torch.float': torch.float,
                'auto': 'auto',
                'None': None
            }.get(model_kwargs['torch_dtype'])
        self.logger.debug(f'HF using torch_dtype: {torch_dtype}')
        if torch_dtype is not None:
            model_kwargs['torch_dtype'] = torch_dtype

    def _load_model(self,
                    path: str,
                    model_kwargs: dict,
                    peft_path: Optional[str] = None):
        from transformers import AutoModel, AutoModelForCausalLM

        self._set_model_kwargs_torch_dtype(model_kwargs)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                path, trust_remote_code = True,**model_kwargs)
        except ValueError:
            self.model = AutoModel.from_pretrained(path, **model_kwargs)

        if peft_path is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model,
                                                   peft_path,
                                                   is_trainable=False)
        self.model.eval()
        self.model.generation_config.do_sample = False

        # A patch for llama when batch_padding = True
        if 'decapoda-research/llama' in path:
            self.model.config.bos_token_id = 1
            self.model.config.eos_token_id = 2
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

    # def generate(self,
    #              inputs: List[str],
    #              max_out_len: int,
    #              min_out_len: Optional[int] = None,
    #              stopping_criteria: List[str] = [],
    #              **kwargs) -> List[str]:
    #     """Generate results given a list of inputs.

    #     Args:
    #         inputs (List[str]): A list of strings.
    #         max_out_len (int): The maximum length of the output.
    #         min_out_len (Optional[int]): The minimum length of the output.

    #     Returns:
    #         List[str]: A list of generated strings.
    #     """
    #     generation_kwargs = kwargs.copy()
    #     generation_kwargs.update(self.generation_kwargs)
    #     if self.batch_padding and len(inputs) > 1:
    #         return self._batch_generate(inputs=inputs,
    #                                     max_out_len=max_out_len,
    #                                     min_out_len=min_out_len,
    #                                     stopping_criteria=stopping_criteria,
    #                                     **generation_kwargs)
    #     else:
    #         return sum(
    #             (self._single_generate(inputs=[input_],
    #                                    max_out_len=max_out_len,
    #                                    min_out_len=min_out_len,
    #                                    stopping_criteria=stopping_criteria,
    #                                    **generation_kwargs)
    #              for input_ in inputs), [])

    def _batch_generate(self,
                        inputs: List[str],
                        max_out_len: int,
                        min_out_len: Optional[int] = None,
                        stopping_criteria: List[str] = [],
                        **kwargs) -> List[str]:
        """Support for batch prompts inference.

        Args:
            inputs (List[str]): A list of strings.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        if self.extract_pred_after_decode:
            prompt_lens = [len(input_) for input_ in inputs]

        if self.use_fastchat_template:
            try:
                from fastchat.model import get_conversation_template
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    'Fastchat is not implemented. You can use '
                    '\'pip install "fschat[model_worker,webui]"\' '
                    'to implement fastchat.')
            for i in range(len(inputs)):
                conv = get_conversation_template('vicuna')
                conv.append_message(conv.roles[0], inputs[i])
                conv.append_message(conv.roles[1], None)
                inputs[i] = conv.get_prompt()

        # step-1: tokenize the input with batch_encode_plus
        tokens = self.tokenizer.batch_encode_plus(inputs,
                                                  padding=True,
                                                  truncation=True,
                                                  max_length=self.max_seq_len)
        tokens = {
            k: torch.tensor(np.array(tokens[k]), device=self.model.device)
            for k in tokens if k in ['input_ids', 'attention_mask']
        }

        origin_stopping_criteria = stopping_criteria
        if stopping_criteria:
            # Construct huggingface stopping criteria
            if self.tokenizer.eos_token is not None:
                stopping_criteria = stopping_criteria + [
                    self.tokenizer.eos_token
                ]
            stopping_criteria = transformers.StoppingCriteriaList([
                *[
                    MultiTokenEOSCriteria(sequence, self.tokenizer,
                                          tokens['input_ids'].shape[0])
                    for sequence in stopping_criteria
                ],
            ])
            kwargs['stopping_criteria'] = stopping_criteria

        if min_out_len is not None:
            kwargs['min_new_tokens'] = min_out_len

        # step-2: conduct model forward to generate output
        outputs = self.model.generate(**tokens,
                                      max_new_tokens=max_out_len,
                                      **kwargs)

        if not self.extract_pred_after_decode:
            outputs = outputs[:, tokens['input_ids'].shape[1]:]

        decodeds = self.tokenizer.batch_decode(outputs,
                                               skip_special_tokens=True)

        if self.extract_pred_after_decode:
            decodeds = [
                token[len_:] for token, len_ in zip(decodeds, prompt_lens)
            ]

        if self.end_str:
            decodeds = [token.split(self.end_str)[0] for token in decodeds]
        if origin_stopping_criteria:
            for t in origin_stopping_criteria:
                decodeds = [token.split(t)[0] for token in decodeds]
        return decodeds

    def _single_generate(self,
                         inputs: List[str],
                         max_out_len: int,
                         min_out_len: Optional[int] = None,
                         stopping_criteria: List[str] = [],
                         **kwargs) -> List[str]:
        """Support for single prompt inference.

        Args:
            inputs (List[str]): A list of strings.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        if self.extract_pred_after_decode:
            prompt_lens = [len(input_) for input_ in inputs]

        if self.use_fastchat_template:
            try:
                from fastchat.model import get_conversation_template
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    'Fastchat is not implemented. You can use '
                    '\'pip install "fschat[model_worker,webui]"\' '
                    'to implement fastchat.')
            conv = get_conversation_template('vicuna')
            conv.append_message(conv.roles[0], inputs[0])
            conv.append_message(conv.roles[1], None)
            inputs = [conv.get_prompt()]

        if self.mode == 'mid':
            input_ids = self.tokenizer(inputs, truncation=False)['input_ids']
            input_ids = torch.tensor(input_ids, device=self.model.device)
            if len(input_ids[0]) > self.max_seq_len - max_out_len:
                half = int((self.max_seq_len - max_out_len) / 2)
                inputs = [
                    self.tokenizer.decode(input_ids[0][:half],
                                          skip_special_tokens=True) +
                    self.tokenizer.decode(input_ids[0][-half:],
                                          skip_special_tokens=True)
                ]

        input_ids = self.tokenizer(inputs,
                                   truncation=True,
                                   max_length=self.max_seq_len -
                                   max_out_len)['input_ids']
        input_ids = torch.tensor(input_ids, device=self.model.device)
        origin_stopping_criteria = stopping_criteria
        if stopping_criteria:
            # Construct huggingface stopping criteria
            if self.tokenizer.eos_token is not None:
                stopping_criteria = stopping_criteria + [
                    self.tokenizer.eos_token
                ]
            stopping_criteria = transformers.StoppingCriteriaList([
                *[
                    MultiTokenEOSCriteria(sequence, self.tokenizer,
                                          input_ids.shape[0])
                    for sequence in stopping_criteria
                ],
            ])
            kwargs['stopping_criteria'] = stopping_criteria

        if min_out_len is not None:
            kwargs['min_new_tokens'] = min_out_len

        # To accommodate the PeftModel, parameters should be passed in
        # key-value format for generate.
        outputs = self.model.generate(input_ids=input_ids,
                                      max_new_tokens=max_out_len,
                                      **kwargs)

        if not self.extract_pred_after_decode:
            outputs = outputs[:, input_ids.shape[1]:]

        decodeds = self.tokenizer.batch_decode(outputs,
                                               skip_special_tokens=True)

        if self.extract_pred_after_decode:
            decodeds = [
                token[len_:] for token, len_ in zip(decodeds, prompt_lens)
            ]

        if self.end_str:
            decodeds = [token.split(self.end_str)[0] for token in decodeds]
        if origin_stopping_criteria:
            for t in origin_stopping_criteria:
                decodeds = [token.split(t)[0] for token in decodeds]
        return decodeds

    # def get_loglikelihood(
    #         self,
    #         inputs: List[str],
    #         conts: List[str],
    #         mask_length: Optional[List[int]] = None) -> List[float]:
    #     """Get loglikelihood scores given a list of inputs.

    #     Args:
    #         inputs (List[str]): A list of strings.
    #         conts (List[str]): A list of strings: slices after the space.
    #         NOT SUPPORT mask_length YET!
    #         mask_length (Optional[List[int]]): A list of mask lengths. If
    #             provided, the perplexity scores will be calculated with the
    #             first mask_length[i] tokens masked out. It's okay to skip
    #             its implementation if advanced features in PPLInfernecer is
    #             not needed.

    #     Returns:
    #         List[float]: A list of loglikelihood scores.
    #     """
    #     assert mask_length is None, 'Not support mask_length yet.'
    #     if self.batch_padding and len(inputs) > 1:
    #         assert self.tokenizer.pad_token
    #         return self._get_loglikelihood(inputs, conts)
    #     else:
    #         return np.concatenate([
    #             self._get_loglikelihood(inputs=[inputs[idx]],
    #                                     conts=[conts[idx]])
    #             for idx in range(len(inputs))
    #         ])

    def _get_loglikelihood(self, inputs: str, conts: str) -> float:
        """Get loglikelihood scores given input string and continuation string.

        Args:
            inputs (str): string.
            conts (str): strings: slices after the space.
        Returns:
            float: loglikelihood scores.
        """
        input_tokenizer_out = self.tokenizer(inputs,
                                             padding=True,
                                             truncation=False,
                                             return_length=True,
                                             return_tensors='pt').to(
                                                 self.model.device)

        input_ids = input_tokenizer_out['input_ids'][:, :self.max_seq_len]
        input_length = input_tokenizer_out['length']
        context_ids = [
            self.tokenizer(inputs[i].replace(conts[i], ''),
                           padding=False,
                           truncation=True,
                           max_length=self.max_seq_len)['input_ids']
            for i in range(len(inputs))
        ]
        # forward
        outputs = self.model(input_ids)['logits']
        outputs = torch.nn.functional.log_softmax(outputs, dim=-1)
        # calculate loglikelihood
        answer = np.zeros(len(inputs))
        for i in range(len(inputs)):
            if self.tokenizer.padding_side == 'right':
                cont_ids = input_ids[i, len(context_ids[i]):input_length[i]]
                logits = outputs[i,
                                 len(context_ids[i]) - 1:input_length[i] -
                                 1, :]  # noqa
            else:
                cont_ids = input_ids[i, len(context_ids[i]) - input_length[i]:]
                logits = outputs[i,
                                 len(context_ids[i]) - input_length[i] - 1:-1]
            # Reducing the dimension will lead to a wrong outcome
            logits_gather = torch.gather(
                logits.unsqueeze(0), 2,
                cont_ids.unsqueeze(0).unsqueeze(-1))  # [1, seq]
            # Answer: sum the likelihood of each token in continuation
            answer[i] = float(logits_gather.detach().cpu().sum())
        return answer

    def get_mink_percent(self, inputs: List[str], k: int = 20) -> List[float]:
        """https://swj0419.github.io/detect-pretrain.github.io/"""

        if self.batch_padding and len(inputs) > 1:
            assert self.tokenizer.pad_token
            return self._get_mink_percent(inputs, k=k)
        else:
            return np.concatenate([
                self._get_mink_percent(inputs=[text], k=k) for text in inputs
            ])

    def _get_mink_percent(self, inputs: List[str], k: int = 20) -> List[float]:
        outputs, inputs = self.get_logits(inputs)
        shift_logits = outputs[:, :-1, :].contiguous().float()
        shift_labels = inputs['tokens']['input_ids'][:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(
            reduction='none', ignore_index=self.tokenizer.pad_token_id)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)).view(shift_labels.size())
        lens = (inputs['tokens']['input_ids'] !=
                self.tokenizer.pad_token_id).sum(-1).cpu().numpy()
        mink_percent = []
        for nloss, nlen in zip(loss, lens):
            nlen = int(nlen)
            minklen = max(nlen * k // 100, 1)
            nloss = torch.topk(loss[-nlen:], minklen, dim=-1)[0]
            nloss = -nloss.float().mean().cpu().detach().numpy()
            mink_percent.append(nloss)
        return np.array(mink_percent)

    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized strings.

        Args:
            prompt (str): Input string.

        Returns:
            int: Length of the input tokens
        """
        return len(self.tokenizer.encode(prompt))

    def prompt_to_3shot_chat(self,prompt_text):
        """
        将prompt转换为3-shot chat格式
        """
        import re
        import json
        messages = []
        
        # 匹配所有任务块
        pattern = r"You are an expert Python programmer, and here is your task:(.*?)\[BEGIN\]\s*'(.*?)'\s*\[DONE\]"
        matches = re.findall(pattern, prompt_text, re.DOTALL)
        
        # 处理前3个作为示例
        # print(len(matches))
        for i, (task, code) in enumerate(matches[:3]):
            # 清理任务描述
            task = task.strip()
            # 清理代码（替换转义字符）
            code = code.replace('\\r\\n', '\n').replace('\\n', '\n')
            # code = '[BEGIN]' + code + '[DONE]'
            messages.append({"role": "user", "content": task})
            messages.append({"role": "assistant", "content": code})
        
        # 处理最后一个任务（如果存在第4个）
        if len(matches) > 3:
            last_task = matches[3][0].strip()
            messages.append({"role": "user", "content": last_task})
        else:
            # 如果没有第4个完整任务，检查是否有未完成的任务
            remaining = prompt_text.split("[DONE]")[-1]
            task_match = re.search(r"You are an expert Python programmer, and here is your task:(.*)", remaining, re.DOTALL)
            if task_match:
                last_task = task_match.group(1).replace("[BEGIN]", "").strip()
                messages.append({"role": "user", "content": last_task})
        
        return messages
    def batch_generate(self, prompt):
        '''
        Args:
            model: Mask predictor.
            prompt: A tensor of shape (B, L), potentially left-padded.
            steps: Sampling steps, less than or equal to gen_length.
            gen_length: Generated answer length.
            block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
            temperature: Categorical distribution sampling temperature.
            cfg_scale: Unsupervised classifier-free guidance scale.
            remasking: Remasking strategy. 'low_confidence' or 'random'.
            mask_id: The token id of [MASK].
            padding_id: The token id for padding.
        '''
        steps = self.gen_steps
        gen_length = self.gen_length
        block_length = self.gen_blocksize
        cfg_scale = self.cfg
        temperature = self.temperature
        remasking = self.remasking
        mask_id = self.mask_id
        padding_id = self.padding_id
        print('parameters:', steps, gen_length, block_length, temperature, cfg_scale, remasking, mask_id)
        messages = _convert_chat_messages(prompt)
        prompt = [self.tokenizer.apply_chat_template(m_i, add_generation_prompt=True, tokenize=False) for m_i in messages]
        print('final prompt:', prompt)
        prompt = self.tokenizer.batch_encode_plus(prompt, padding = True, return_tensors='pt')['input_ids']
        # prompt = self.tokenizer(prompt,return_tensors='pt')['input_ids']
        batch_size, prompt_len = prompt.shape
        total_len = prompt_len + gen_length
        
        # 1. 初始化输入序列x，prompt部分填充真实token，生成部分填充[MASK]
        is_left = False
        for i in range(batch_size):
            if (prompt[i, 0] == padding_id) and (prompt[i, -1] != padding_id):
                is_left = True
                break
        if not is_left:
            batch_padding_position = []
            for i in range(batch_size):
                padding_pos = (prompt[i] == padding_id).nonzero(as_tuple=True)[0]
                if len(padding_pos) > 0:
                    batch_padding_position.append(padding_pos[0].item())
                else:
                    batch_padding_position.append(prompt_len)
            
            ## where padding start
            prompt_len = batch_padding_position.copy()

        x = torch.full((batch_size, total_len), mask_id, dtype=torch.long, device=self.model.device)
        if not is_left:
            for i in range(batch_size):
                x[i, :prompt_len[i]] = prompt[i, :prompt_len[i]].clone()
                x[i, prompt_len[i] + gen_length:] = torch.full((total_len - (prompt_len[i] + gen_length),), padding_id, dtype=torch.long) 
        else:
            x[:, :prompt_len] = prompt.clone()

        # 2. 创建一个布尔掩码，标记出原始prompt中非padding的位置
        original_prompt_mask = (prompt != padding_id)
        # 创建一个在整个序列长度上的布尔掩码，用于CFG
        prompt_index = torch.zeros_like(x, dtype=torch.bool)
        if not is_left:
            for i in range(batch_size):
                prompt_index[i, :prompt_len[i]] = original_prompt_mask[i, :prompt_len[i]]
        else:
            prompt_index[:, :prompt_len] = original_prompt_mask

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length

        assert steps % num_blocks == 0
        steps_per_block = steps // num_blocks
        
        # --- 修复开始 ---
        # 3. 正确构建 attention_mask
        # 创建一个总长度的 attention_mask，默认所有位置都可被关注 (值为1)
        attention_mask = torch.ones((batch_size, total_len), device=x.device, dtype=torch.long)
        # 仅将原始prompt中的padding位置设置为0，这样无论左padding还是右padding都能正确处理
        if not is_left:
            for i in range(batch_size):
                # attention_mask[i, :prompt_len[i]+gen_length] = original_prompt_mask[i, :prompt_len[i]+gen_length].long()
                attention_mask[i, prompt_len[i] + gen_length:] = 0  # 生成部分之后的padding位置设为0
        else:
            attention_mask[:, :prompt_len] = original_prompt_mask.long()
        for num_block in range(num_blocks):
            # 注意：这里的 block_mask_index 和 num_transfer_tokens 的逻辑需要您根据模型具体实现来确认
            # 假设 get_num_transfer_tokens 是一个已定义的辅助函数
            if not is_left:
                start_pos = [prompt_len[i] + num_block * block_length for i in range(batch_size)]
                end_pos = [prompt_len[i] + (num_block + 1) * block_length for i in range(batch_size)]
                block_mask_index = torch.zeros((batch_size, block_length), dtype=torch.bool, device=x.device)
                for i in range(batch_size):
                    block_mask_index[i] = (x[i, start_pos[i]:end_pos[i]] == mask_id)
            else:
                start_pos = prompt_len + num_block * block_length
                end_pos = prompt_len + (num_block + 1) * block_length
                block_mask_index = (x[:, start_pos:end_pos] == mask_id)
            
            # 假设这个函数返回每个step要替换的token数量
            # 注意：此函数的实现细节可能会影响生成效果
            # num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
            # print((attention_mask != 1).sum())
            for i in range(steps_per_block):
                mask_index = (x == mask_id)
                if not mask_index.any(): # 如果没有mask了，提前结束
                    break
                    
                if cfg_scale > 0.:
                    un_x = x.clone()
                    # 在CFG的unconditional输入中，屏蔽掉真实的prompt
                    un_x[prompt_index] = mask_id
                    
                    # 将conditional和unconditional输入拼接
                    x_ = torch.cat([x, un_x], dim=0)
                    # attention_mask也需要相应地拼接
                    extended_attention_mask = torch.cat([attention_mask, attention_mask], dim=0)
                    
                    logits = self.model(x_, attention_mask=extended_attention_mask).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self.model(x, attention_mask=attention_mask).logits
                    max_logits = torch.argmax(logits, dim=-1)
                    # print("max logit 0:", max_logits[0][mask_index[0]])
                    # print("max logit 1:", max_logits[-1][mask_index[-1]])
                    # print("logits shape:", logits.shape)
                    # print(logits[-1])

                # --- 后续的采样和重掩码逻辑保持不变 ---
                if temperature > 0.:
                    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                    x0 = torch.argmax(logits_with_noise, dim=-1)
                else:
                    x0 = torch.argmax(logits, dim=-1)

                if remasking == 'low_confidence':
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(-1)
                elif remasking == 'random':
                    x0_p = torch.rand_like(logits[:, :, 0])
                else:
                    raise NotImplementedError(remasking)

                # 只在当前block及之前的位置计算置信度
                confidence_mask = torch.ones_like(x, dtype=torch.bool)
                if not is_left:
                    for j in range(batch_size):
                        confidence_mask[j, end_pos[j]:] = False
                else:
                    confidence_mask[:, end_pos:] = False
                
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index & confidence_mask, x0_p, -torch.inf)

                # 决定在这一步揭示多少个token
                # 这里的逻辑可能需要根据您的具体算法调整，以下是一个示例
                num_masked_in_block = block_mask_index.sum(dim=-1)
                num_to_reveal = torch.ceil(num_masked_in_block * (1.0 / steps_per_block)).long()

                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(confidence.shape[0]):
                    k = min(num_to_reveal[j].item(), (confidence[j] > -torch.inf).sum().item())
                    if k > 0:
                        _, select_index = torch.topk(confidence[j], k=k)
                        transfer_index[j, select_index] = True
                
                x[transfer_index] = x0[transfer_index]
                true_indices = torch.where(transfer_index[-1])[0].tolist()
                # print(mask_index[-1])
                # print(f'True indices: {true_indices}')
                # print(f'New tokens: {x0[-1, true_indices]}')
        if not is_left:
            for i in range(batch_size):
                x_generated = x[i, prompt_len[i]:prompt_len[i] + gen_length].clone()
                x[i,-gen_length:] = x_generated.clone()
                x[i, : -gen_length] = prompt[i].clone()
                # x[i, : prompt_len[i]] = prompt[i, :prompt_len[i]].clone()
        # response = self.tokenizer.batch_decode(x[:, -gen_length:], skip_special_tokens=True)
        responses = []
        for i in range(batch_size):
            responses.append(self.tokenizer.decode(x[i, -gen_length:], skip_special_tokens=True))
        print('--------------------')
        for i in range(batch_size):
            print(f'Response {i}:', responses[i])
            print('====================')
        print('--------------------')
        return responses
    @ torch.no_grad()
    def _generate(self, prompt):
        '''
        Args:
            model: Mask predictor.
            prompt: A tensor of shape (1, L).
            steps: Sampling steps, less than or equal to gen_length.
            gen_length: Generated answer length.
            block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
            temperature: Categorical distribution sampling temperature.
            cfg_scale: Unsupervised classifier-free guidance scale.
            remasking: Remasking strategy. 'low_confidence' or 'random'.
            mask_id: The toke id of [MASK] is 126336.
        '''
        print('--------------------')
        steps = self.gen_steps
        gen_length = self.gen_length
        block_length = self.gen_blocksize
        cfg_scale = self.cfg
        temperature = self.temperature
        remasking = self.remasking
        mask_id = self.mask_id
        print('parameters:', steps, gen_length, block_length, temperature, cfg_scale, remasking, mask_id)
        # print('prompt:', prompt)
        # print(type(prompt[0]))
        # messages = [{"role": "user", "content": prompt[0]}]
        messages = _convert_chat_messages(prompt)
        # m = self.prompt_to_3shot_chat(prompt[0])
        # print(messages)
        # prompt = self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        prompt = [self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in messages]
        print('final prompt:', prompt)
        prompt = self.tokenizer(prompt,return_tensors='pt')['input_ids']
        # print(prompt.shape)
        # print(prompt)
        # print(type(prompt[0]))
        x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(self.model.device)
        # print(prompt.shape)
        x[:, :prompt.shape[1]] = prompt.clone()

        prompt_index = (x != mask_id)

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length

        assert steps % num_blocks == 0
        steps = steps // num_blocks

        for num_block in range(num_blocks):
            block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
            # print(num_transfer_tokens.sum())
            for i in range(steps):
                mask_index = (x == mask_id)
                if cfg_scale > 0.:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self.model(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self.model(x).logits

                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

                if remasking == 'low_confidence':
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
                elif remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                    transfer_index[j, select_index] = True
                x[transfer_index] = x0[transfer_index]
        response = self.tokenizer.batch_decode(x[:, prompt.shape[1]:], skip_special_tokens=True)
        print('--------------------')
        print('response:', response)
        print('--------------------')
        return response

    def generate(self, inputs: List[str], max_out_len: int) -> List[str]:
        """Generate results given a list of inputs. """
        batch_size = len(inputs)
        if batch_size > 1:
            # assert batch_size == self.batch_size, f"Current batch_size should be equal to the predefined batch_size, {self.batch_size}, but got {batch_size}."
            return self.batch_generate(inputs)
        return self._generate(inputs)
    
    def get_ppl(self,
                inputs: List[str],
                mask_length: Optional[List[int]] = None) -> List[float]:
        """Get perplexity scores given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            mask_length (Optional[List[int]]): A list of mask lengths. If
                provided, the perplexity scores will be calculated with the
                first mask_length[i] tokens masked out. It's okay to skip
                its implementation if advanced features in PPLInfernecer is
                not needed.

        Returns:
            List[float]: A list of perplexity scores.
        """

        if self.batch_padding and len(inputs) > 1:
            assert self.tokenizer.pad_token
            return self._get_ppl(inputs, mask_length=mask_length)
        else:
            if mask_length is not None:
                print('_______')
                return np.concatenate([
                    self._get_ppl(inputs=[text],
                                         mask_length=[mask_length[idx]])
                    for idx, text in enumerate(inputs)
                ])
            return np.concatenate([
                self._get_ppl(inputs=[text], mask_length=mask_length)
                for text in inputs
            ])
    def _get_ppl_masked(self, inputs: List[str], mask_length: Optional[List[int]] = None) -> List[float]:
        """Get perplexity scores given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            mask_length (Optional[List[int]]): A list of mask lengths. If
                provided, the perplexity scores will be calculated with the
                first mask_length[i] tokens masked out. It's okay to skip
                its implementation if advanced features in PPLInfernecer is
                not needed.

        Returns:
            List[float]: A list of perplexity scores.
        """
        
        # 1. Tokenize输入
        # print(repr(inputs[0]))
        print(inputs)
        prefix_index = len(inputs[0]) - len(mask_length[0])
        mask_length = None
        tokenized = self.tokenizer(
            inputs, 
            return_tensors='pt'
        ).to(self.model.device)
        prefix = self.tokenizer(inputs[0][:prefix_index], return_tensors='pt').to(self.model.device)
        batch = tokenized['input_ids']
        # print(len(batch[0]),len(prefix['input_ids'][0]))
        # 2. 计算prompt_index
        if mask_length is not None:
            max_mask_len = max(mask_length)
            prompt_index = torch.arange(batch.shape[1], device=batch.device) < max_mask_len
        else:
            # mask_length = [batch.shape[1] - 1]
            # max_mask_len = batch.shape[1] - 1

            max_mask_len = len(prefix['input_ids'][0])
            mask_length = [max_mask_len]
            prompt_index = torch.arange(batch.shape[1], device=batch.device) < max_mask_len
            # prompt_index = torch.zeros(batch.shape[1], dtype=torch.bool, device=batch.device)
        
        # 3. 为每个样本单独计算困惑度
        single_seq = batch.repeat(self.batch_size, 1)
        # 计算target部分长度（用于归一化）
        lens = single_seq.shape[1]
        target_len = (single_seq.shape[1] - mask_length[0])
        # target_len = (batch.shape[1] - max_mask_len)
        # print('----------------')
        # print(target_len)
        
        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            # 使用扩散前向过程创建掩码版本
            perturbed_seq, p_mask = self._forward_process(single_seq, prompt_index)
            
            # 找到被mask的位置
            mask_indices = perturbed_seq == self.mask_id
            
            # 获取模型预测
            logits = self.get_logits(perturbed_seq, prompt_index)
            
            # 计算掩码位置的交叉熵损失，应用重要性采样权重
            loss = F.cross_entropy(
                logits[mask_indices], 
                single_seq[mask_indices], 
                reduction='none'
            ) / p_mask[mask_indices]
            
            normalized_loss = loss.sum() / (self.batch_size * target_len)
            # all lens
            # normalized_loss = loss.sum() / (lens * self.batch_size)
            # normalized_loss = normalized_loss / target_len
            loss_acc.append(normalized_loss.item())
        
        # 对所有蒙特卡洛样本取平均
        sample_ppl = sum(loss_acc) / len(loss_acc)
        sample_ppl = [sample_ppl]
        return sample_ppl

    def _forward_process(self, batch, prompt_index):
        b, l = batch.shape

        target_len = (l - prompt_index.sum()).item()
        k = torch.randint(1, target_len + 1, (), device=batch.device)

        x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
        x = ((x - 1) % target_len) + 1
        assert x.min() >= 1 and x.max() <= target_len

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)

        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)

        noisy_batch = torch.where(is_mask, self.mask_id, batch)

        return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)
    def get_answer_prefix_index(self, text: str) -> int:
        """获取答案部分的起始索引"""
        
        # 尝试不同的答案模式
        patterns = [
            ('Answer:', 'Answer:'),
            ('Answer: \n', 'Answer: '),  # 移除换行符
            ('答案:', '答案:'),
            ('答案是:', '答案是:'),
            ('A:', 'A:'),
        ]
        
        for pattern, replacement in patterns:
            index = text.rfind(pattern)
            if index != -1:
                if pattern == 'Answer: \n':
                    # 特殊处理：替换换行符
                    text = text.replace('Answer: \n', 'Answer: ')
                    return text.rfind('Answer:') + len('Answer:')
                else:
                    return index + len(pattern)
        
        # 如果都没找到，使用最后一个换行符（hellaswag的情况）
        print('this is target for hellaswag')
        last_newline = text.rfind('\n')
        return last_newline +1 if last_newline != -1 else 0

    def get_loglikelihood(self, inputs: List[str], conts:  List[str]) -> List[float]:
        print('inputs:', inputs, 'conts:', conts)
        mask_length = [self.get_token_len(c, add_special_tokens=False) for c in conts]
        return - self.get_ppl(inputs, mask_length)
    def _get_ppl(self,
            inputs: List[str],
            mask_length: Optional[List[int]] = None) -> List[float]:
        """Get perplexity scores using diffusion-based Monte Carlo sampling."""
        
        # 1. Tokenize输入
        # print(repr(inputs[0]))
        print(inputs)
        print(mask_length)
        prefix_index = 0
        if mask_length is None:
            inputs[0] = inputs[0].replace('Answer: \n', 'Answer: ')
            prefix_index = self.get_answer_prefix_index(inputs[0])
        ## hellaswag
            inputs[0] = inputs[0][:prefix_index] + ' ' + inputs[0][prefix_index+1:]
        # prefix_index = inputs[0].rfind('Answer:') + len('Answer:') ##arcc 'Answer: \n' 35.x, 'Answer: ' 41.x, 'Answer:' 40.x
        # if prefix_index == -1 + len('Answer:'):
        #     prefix_index = inputs[0].rfind('答案:') + len('答案: \n') ## ceval
        # elif prefix_index == -1 + len('答案: \n'):
        #     prefix_index = inputs[0].rfind('答案是:') + len('答案是:') ## cmmlu. 不知道为什么这里不能 \n
        # elif prefix_index == -1 + len('答案是:'):
        #     prefix_index = inputs[0].rfind('A:') + len('A:')
        # elif prefix_index == -1 + len('A:'):
        #     print('this is target for hellaswag')
        #     prefix_index = inputs[0].rfind('\n') + len('\n')
        # print(inputs[0][prefix_index:])
        # print(repr(inputs[0]))
        tokenized = self.tokenizer(
            inputs, 
            return_tensors='pt'
        ).to(self.model.device)
        if mask_length is None:
            prefix = self.tokenizer(inputs[0][:prefix_index], return_tensors='pt').to(self.model.device)
        batch = tokenized['input_ids']
        # print(len(batch[0]),len(prefix['input_ids'][0]))
        # 2. 计算prompt_index
        if mask_length is not None:
            max_mask_len = batch.shape[1] - max(mask_length)
            prompt_index = torch.arange(batch.shape[1], device=batch.device) < max_mask_len
        else:
            # mask_length = [batch.shape[1] - 1]
            # max_mask_len = batch.shape[1] - 1

            max_mask_len = len(prefix['input_ids'][0])
            mask_length = [max_mask_len]
            prompt_index = torch.arange(batch.shape[1], device=batch.device) < max_mask_len
            # prompt_index = torch.zeros(batch.shape[1], dtype=torch.bool, device=batch.device)
        
        # 3. 为每个样本单独计算困惑度
        single_seq = batch.repeat(self.batch_size, 1)
        # 计算target部分长度（用于归一化）
        lens = single_seq.shape[1]
        target_len = (single_seq.shape[1] - mask_length[0])
        # target_len = (batch.shape[1] - max_mask_len)
        # print('----------------')
        # print(target_len)
        
        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            # 使用扩散前向过程创建掩码版本
            perturbed_seq, p_mask = self._forward_process(single_seq, prompt_index)
            
            # 找到被mask的位置
            mask_indices = perturbed_seq == self.mask_id
            
            # 获取模型预测
            logits = self.get_logits(perturbed_seq, prompt_index)
            
            # 计算掩码位置的交叉熵损失，应用重要性采样权重
            loss = F.cross_entropy(
                logits[mask_indices], 
                single_seq[mask_indices], 
                reduction='none'
            ) / p_mask[mask_indices]
            
            normalized_loss = loss.sum() / (self.batch_size * target_len)
            # all lens
            # normalized_loss = loss.sum() / (lens * self.batch_size)
            # normalized_loss = normalized_loss / target_len
            loss_acc.append(normalized_loss.item())
        
        # 对所有蒙特卡洛样本取平均
        sample_ppl = sum(loss_acc) / len(loss_acc)
        sample_ppl = [sample_ppl]
        return sample_ppl
    def get_logits(self, inputs, prompt_index):
        if self.cfg > 0.:
            assert len(prompt_index) == inputs.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(inputs.shape[0], 1)
            un_inputs = inputs.clone()
            un_inputs[prompt_index] = self.mask_id
            inputs = torch.cat([inputs, un_inputs])

        logits = self.model(inputs).logits

        if self.cfg > 0.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)
        return logits[:, :inputs.shape[1]]

class LLaDABaseModel(LLaDAModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def _generate(self, prompt,max_out_len = 1024,min_out_len =1024,stopping_criteria=[]):
        '''
        Args:
            model: Mask predictor.
            prompt: A tensor of shape (1, L).
            steps: Sampling steps, less than or equal to gen_length.
            gen_length: Generated answer length.
            block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
            temperature: Categorical distribution sampling temperature.
            cfg_scale: Unsupervised classifier-free guidance scale.
            remasking: Remasking strategy. 'low_confidence' or 'random'.
            mask_id: The toke id of [MASK] is 126336.
        '''
        print('--------------------')
        steps = self.gen_steps
        gen_length = self.gen_length
        block_length = self.gen_blocksize
        cfg_scale = self.cfg
        temperature = self.temperature
        remasking = self.remasking
        mask_id = self.mask_id
        stopping_criteria = set(stopping_criteria+self.stop_words)
        print('parameters:', steps, gen_length, block_length, temperature, cfg_scale, remasking, mask_id, stopping_criteria)

        messages = _convert_base_messages(prompt)
        print('final prompt:', messages)
        prompt = self.tokenizer(messages)["input_ids"]
        prompt = torch.tensor(prompt, dtype=torch.long, device=self.model.device)
        # print(prompt.shape)
        # print(prompt)
        # print(type(prompt[0]))
        x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(self.model.device)
        # print(prompt.shape)
        x[:, :prompt.shape[1]] = prompt.clone()

        prompt_index = (x != mask_id)

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length

        assert steps % num_blocks == 0
        steps = steps // num_blocks

        for num_block in range(num_blocks):
            block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
            # print(num_transfer_tokens.sum())
            for i in range(steps):
                mask_index = (x == mask_id)
                if cfg_scale > 0.:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self.model(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self.model(x).logits

                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

                if remasking == 'low_confidence':
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
                elif remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                    transfer_index[j, select_index] = True
                x[transfer_index] = x0[transfer_index]
        response = self.tokenizer.batch_decode(x[:, prompt.shape[1]:], skip_special_tokens=True)
        response = response[0]
        for stop_word in stopping_criteria:
            if stop_word in response:
                response = response.split(stop_word)[0]
                break
        response = [response]
        print('--------------------')
        print('response:', response)
        print('--------------------')
        return response
    def get_token_len(self, prompt: str, add_special_tokens: bool=True) -> int:
        m = _convert_base_messages([prompt])[0]
        t = self.tokenizer(m, add_special_tokens=add_special_tokens)
        return len(t['input_ids'])
##########
# flake8: noqa
# yapf: disable
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Union

from opencompass.models.base import BaseModel, LMTemplateParser
from opencompass.models.base_api import APITemplateParser
from opencompass.registry import MODELS
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList

PromptType = Union[PromptList, str]


def _get_stopping_criteria(stop_words, tokenizer, batch_size):
    from transformers import StoppingCriteria, StoppingCriteriaList

    class MultiTokenEOSCriteria(StoppingCriteria):
        """Criteria to stop on the specified multi-token sequence."""

        def __init__(self, stop_words: List[str], tokenizer, batch_size: int):
            self.done_tracker = [False] * batch_size
            self.stop_words, self.max_sequence_id_len = [], 0
            for s in stop_words:
                self.stop_words.append(s)
                sequence_ids = tokenizer.encode(s, add_special_tokens=False)
                self.max_sequence_id_len = max(self.max_sequence_id_len, len(sequence_ids))
            self.tokenizer = tokenizer

        def __call__(self, input_ids, scores, **kwargs) -> bool:
            # compare the last len(stop) tokens
            lookback_ids_batch = input_ids[:, -self.max_sequence_id_len:]
            lookback_tokens_batch = self.tokenizer.batch_decode(lookback_ids_batch)
            for i, done in enumerate(self.done_tracker):
                if done:
                    continue
                self.done_tracker[i] = any(s in lookback_tokens_batch[i] for s in self.stop_words)
            return False not in self.done_tracker

    c = MultiTokenEOSCriteria(stop_words, tokenizer, batch_size)
    return StoppingCriteriaList([c])

def _get_possible_max_seq_len(max_seq_len, path):
    if max_seq_len is not None:
        return max_seq_len

    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    possible_keys = [
        'max_position_embeddings',
        'seq_length',
        'model_max_length',
        'max_sequence_length',
    ]
    for k in possible_keys:
        if hasattr(config, k):
            return getattr(config, k)
    raise ValueError('max_seq_len is not provided and cannot be inferred from the model config.')


def _convert_chat_messages(inputs, merge_role=True, skip_empty_prompt=True):
    outputs = []
    for _input in inputs:
        messages = []
        if isinstance(_input, str):
            messages.append({'role': 'user', 'content': _input})
        else:
            for item in _input:
                if skip_empty_prompt and not item['prompt']:
                    continue
                role = {
                    'HUMAN': 'user',
                    'BOT': 'assistant',
                    'SYSTEM': 'system',
                }[item['role']]
                messages.append({'role': role, 'content': item['prompt']})

        if merge_role:
            merged_messages = []
            for item in messages:
                if merged_messages and merged_messages[-1]['role'] == item['role']:
                    merged_messages[-1]['content'] += '\n' + item['content']
                else:
                    merged_messages.append(item)
            messages = merged_messages

        outputs.append(messages)
    return outputs


def _format_with_fast_chat_template(inputs: List[str], name: str='vicuna'):
    try:
        from fastchat.model import get_conversation_template
    except ImportError:
        raise ModuleNotFoundError('fastchat not found. Please install with\npip install "fschat[model_worker,webui]"')

    outputs = []
    for _input in inputs:
        template = get_conversation_template(name)
        for item in _input:
            if item['role'] == 'user':
                template.append_message(template.roles[0], item['content'])
            elif item['role'] == 'assistant':
                template.append_message(template.roles[1], item['content'])
            elif item['role'] == 'system':
                continue
            else:
                raise ValueError(f'Unknown role {item["role"]}')
        template.append_message(template.roles[1], None)
        outputs.append(template.get_prompt())
    return outputs


def _get_meta_template(meta_template):
    default_meta_template = dict(
        round=[
            dict(role='HUMAN', api_role='HUMAN'),
            # XXX: all system roles are mapped to human in purpose
            # dict(role='SYSTEM', api_role='HUMAN'),
            dict(role='BOT', api_role='BOT', generate=True),
        ],
        reserved_roles=[dict(role='SYSTEM', api_role='SYSTEM')]
    )
    return APITemplateParser(meta_template or default_meta_template)


def _set_model_kwargs_torch_dtype(model_kwargs):
    import torch
    if 'torch_dtype' not in model_kwargs:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = {
            'torch.float16': torch.float16,
            'torch.bfloat16': torch.bfloat16,
            'torch.float': torch.float,
            'auto': 'auto',
            'None': None,
        }.get(model_kwargs['torch_dtype'])
    if torch_dtype is not None:
        model_kwargs['torch_dtype'] = torch_dtype
    return model_kwargs


def add_gumbel_noise(logits, temperature):
    '''
    As suggested by https://arxiv.org/pdf/2409.02908, we use float64 for the gumbel max method.
    '''
    if temperature < 0.01:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


@MODELS.register_module()
class LLaDAwithChatTemplate(BaseModel):

    def __init__(self,
                 path: str,
                 model_kwargs: dict = dict(),
                 tokenizer_path: Optional[str] = None,
                 tokenizer_kwargs: dict = dict(),
                 peft_path: Optional[str] = None,
                 peft_kwargs: dict = dict(),
                 tokenizer_only: bool = False,
                 generation_kwargs: dict = dict(),
                 max_seq_len: Optional[int] = None,
                 meta_template: Optional[Dict] = None,
                 pad_token_id: Optional[int] = None,
                 fastchat_template: Optional[str] = None,
                 stop_words: Optional[str] = [],
                 mask_token_ids: int = 126336,
                 gen_method: str = "",
                 remask_confidence: str = "probability",
                 cfg_scale: float = 0.,
                 add_gen_length: int = 4,
                 add_gen_total_length: int = -1,
                 gen_steps: int = 4,
                 gen_temp_x0: float = 0.,
                 gen_temp_remask: float = 0.,
                 **other_kwargs):

        self.logger = get_logger()
        self.path = path
        self.tokenizer_only = tokenizer_only
        self.template_parser = _get_meta_template(meta_template)
        self.max_seq_len = _get_possible_max_seq_len(max_seq_len, path)
        self._load_tokenizer(tokenizer_path or path, tokenizer_kwargs, pad_token_id)
        if not tokenizer_only:
            self._load_model(path=path, kwargs=model_kwargs, peft_path=peft_path, peft_kwargs=peft_kwargs)
        self.generation_kwargs = generation_kwargs
        self.fastchat_template = fastchat_template
        self.stop_words = list(set(stop_words + self._get_potential_stop_words(path)))
        self.logger.info(f'using stop words: {self.stop_words}')

        self.mask_token_ids = mask_token_ids
        self.gen_method = gen_method
        self.remask_confidence = remask_confidence
        self.cfg_scale = cfg_scale
        self.add_gen_length = add_gen_length
        self.add_gen_total_length = add_gen_total_length
        self.gen_steps = gen_steps
        self.gen_temp_x0 = gen_temp_x0
        self.gen_temp_remask = gen_temp_remask

        for k, v in other_kwargs.items():
            if v is not None:
                self.logger.warning(f'Unused argument {k}={v}')

    def _load_tokenizer(self, path: Optional[str], kwargs: dict, pad_token_id: Optional[int] = None):
        from transformers import AutoTokenizer, GenerationConfig

        DEFAULT_TOKENIZER_KWARGS = dict(padding_side='left', truncation_side='left', trust_remote_code=True)
        tokenizer_kwargs = DEFAULT_TOKENIZER_KWARGS
        tokenizer_kwargs.update(kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(path, **tokenizer_kwargs)

        # A patch for some models without pad_token_id
        if pad_token_id is not None:
            if self.tokenizer.pad_token_id is None:
                self.logger.debug(f'Using {pad_token_id} as pad_token_id')
            elif self.tokenizer.pad_token_id != pad_token_id:
                self.logger.warning(f'pad_token_id is not consistent. Using {pad_token_id} as pad_token_id')
            self.tokenizer.pad_token_id = pad_token_id
            return
        if self.tokenizer.pad_token_id is not None:
            return
        self.logger.warning('pad_token_id is not set for the tokenizer.')
        generation_config = GenerationConfig.from_pretrained(path)
        if generation_config.pad_token_id is not None:
            self.logger.warning(f'Using {generation_config.pad_token_id} as pad_token_id.')
            self.tokenizer.pad_token_id = generation_config.pad_token_id
            return
        if self.tokenizer.eos_token_id is not None:
            self.logger.warning(f'Using eos_token_id {self.tokenizer.eos_token_id} as pad_token_id.')
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            return
        raise ValueError('pad_token_id is not set for this tokenizer. Please set `pad_token_id={PAD_TOKEN_ID}` in model_cfg.')

    def _load_model(self, path: str, kwargs: dict, peft_path: Optional[str] = None, peft_kwargs: dict = dict()):
        from transformers import AutoModel, AutoModelForCausalLM

        DEFAULT_MODEL_KWARGS = dict(device_map='auto', trust_remote_code=True)
        model_kwargs = DEFAULT_MODEL_KWARGS
        model_kwargs.update(kwargs)
        model_kwargs = _set_model_kwargs_torch_dtype(model_kwargs)
        self.logger.debug(f'using model_kwargs: {model_kwargs}')

        try:
            self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        except ValueError:
            self.model = AutoModel.from_pretrained(path, **model_kwargs)

        if peft_path is not None:
            from peft import PeftModel
            peft_kwargs['is_trainable'] = False
            self.model = PeftModel.from_pretrained(self.model, peft_path, **peft_kwargs)

        self.model.eval()
        self.model.generation_config.do_sample = False

    def _get_potential_stop_words(self, path: Optional[str]):
        from transformers import GenerationConfig
        potential_stop_words = []
        try:
            generation_config = GenerationConfig.from_pretrained(path)
        except:
            generation_config = None
        if generation_config and hasattr(generation_config, 'eos_token_id'):
            if isinstance(generation_config.eos_token_id, int):
                potential_stop_words.append(self.tokenizer.decode(generation_config.eos_token_id))
            else:
                assert isinstance(generation_config.eos_token_id, list)
                for token_id in generation_config.eos_token_id:
                    potential_stop_words.append(self.tokenizer.decode(token_id))
        if self.tokenizer.eos_token is not None:
            potential_stop_words.append(self.tokenizer.eos_token)
        potential_stop_words = list(set(potential_stop_words))
        potential_stop_words = [s for s in potential_stop_words if s]
        return potential_stop_words

    def get_logits(self, batch, prompt_index, input_length=-1):
        assert batch.shape[0] == 1
        if self.cfg_scale > 0.:
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            uncond_batch = batch.clone()
            uncond_batch[prompt_index] = self.mask_token_ids
            batch = torch.cat([batch, uncond_batch])

        if input_length > 0:
            assert input_length >= batch.shape[1]
            padding_length = input_length - batch.shape[1]
            padding = torch.full((batch.size(0), padding_length), self.mask_token_ids, device=self.model.device)
            logits = self.model(torch.cat([batch, padding], dim=1)).logits
            logits = logits[:, :batch.shape[1]]
        else:
            logits = self.model(batch).logits
        if self.cfg_scale > 0.:
            logits, uncond_logits = torch.chunk(logits, 2, dim=0)
            logits = uncond_logits + (self.cfg_scale + 1) * (logits - uncond_logits)
        return logits


    def get_transfer_tokens(self):
        # 我们把self.add_gen_length尽可能均匀的分为self.gen_steps份
        base_size = self.add_gen_length // self.gen_steps
        remainder = self.add_gen_length % self.gen_steps
        transfer_tokens = [base_size] * (self.gen_steps - remainder) + [base_size + 1] * remainder
        return transfer_tokens


    def gen_per_steps(self, seq, prompt_index, transfer_token, input_length=-1):
        mask_index = (seq == self.mask_token_ids)
        logits = self.get_logits(seq, prompt_index, input_length)[mask_index]
        logits_with_noise = add_gumbel_noise(logits, self.gen_temp_x0)
        x0 = torch.argmax(logits_with_noise, dim=-1)

        if self.remask_confidence == 'probability':
            log_p = F.log_softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(log_p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(dim=-1)
        elif self.remask_confidence == 'certainty':
            p = F.softmax(logits.to(torch.float64), dim=-1)
            values, _ = torch.topk(logits, k=2, dim=-1)
            confidence = - torch.abs(values[..., 0] - values[..., 1])
        else:
            raise NotImplementedError(self.remask_confidence)
            
        confidence_with_noise = add_gumbel_noise(confidence, self.gen_temp_remask)
        _, index = torch.sort(confidence_with_noise, descending=True)
        x0[index[transfer_token:]] = self.mask_token_ids
        seq[mask_index] = x0.clone()
        return seq


    def generate(self,
                 inputs: List[str],
                 max_out_len: int,
                 min_out_len: Optional[int] = None,
                 stopping_criteria: List[str] = [],
                 **kwargs) -> List[str]:
        messages = _convert_chat_messages(inputs)
        # print(self.fastchat_template)
        if self.fastchat_template:
            messages = _format_with_fast_chat_template(messages, self.fastchat_template)
        else:
            messages = [self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in messages]
        batch_size = len(messages)
        assert batch_size == 1
        print(messages)
        print('--------------------------')

        prompt = self.tokenizer(messages)["input_ids"]
        prompt = torch.tensor(prompt, dtype=torch.long, device=self.model.device)
        print(prompt.shape)
        stopping_criteria = list(set(stopping_criteria + self.stop_words))
        if '<|endoftext|>' in stopping_criteria:
            stopping_criteria.remove('<|endoftext|>')
        if '<|end_of_text|>' in stopping_criteria:
            stopping_criteria.remove('<|end_of_text|>')
        print(stopping_criteria)
        print('^^^^^^^^^^^^^^^^^^^^^^^^^')

        if self.gen_method == 's_ar':
            # step-2: conduct model forward to generate output
            gen_total_length = self.add_gen_total_length if self.add_gen_total_length > 0 else max_out_len
            if gen_total_length % self.add_gen_length == 0:
                num_steps = gen_total_length // self.add_gen_length
            else:
                num_steps = gen_total_length // self.add_gen_length + 1

            seq = prompt.clone()
            stop_generate = False
            input_length = prompt.shape[1] + self.add_gen_total_length if self.add_gen_total_length > 0 else -1
            for _ in range(num_steps):
                mask_seq = torch.full((seq.size(0), self.add_gen_length), self.mask_token_ids, device=self.model.device)
                seq = torch.cat([seq, mask_seq], dim=-1)
                prompt_index = torch.arange(seq.shape[1], device=self.model.device) < prompt.shape[1]
                transfer_tokens = self.get_transfer_tokens()
                for transfer_token in transfer_tokens:
                    seq = self.gen_per_steps(seq, prompt_index, transfer_token, input_length)

                generated_answer = self.tokenizer.decode(seq[0][prompt.shape[1]:], skip_special_tokens=False)
                for stop_seq in stopping_criteria:
                    if stop_seq in generated_answer:
                        stop_generate = True
                        generated_answer = generated_answer.split(stop_seq)[0]

                if stop_generate:
                    break

        elif self.gen_method == 'diff':
            seq = torch.full((1, prompt.shape[1] + self.add_gen_length), self.mask_token_ids, device=self.model.device)
            prompt_index = torch.arange(seq.shape[1], device=self.model.device) < prompt.shape[1]
            seq[0, :prompt.shape[1]] = prompt[0]
            
            transfer_tokens = self.get_transfer_tokens()
            for transfer_token in transfer_tokens:
                seq = self.gen_per_steps(seq, prompt_index, transfer_token)

            generated_answer = self.tokenizer.decode(seq[0][prompt.shape[1]:], skip_special_tokens=False)
            for stop_seq in stopping_criteria:
                if stop_seq in generated_answer:
                    generated_answer = generated_answer.split(stop_seq)[0]
        else:
            raise NotImplementedError(self.gen_method)

        # remove special tokens
        decodeds_ids = self.tokenizer(generated_answer)["input_ids"]
        decodeds = self.tokenizer.decode(decodeds_ids, skip_special_tokens=True)
        print(decodeds)
        print('*****************************************************')
        return [decodeds]

    def get_token_len(self, prompt: str) -> int:
        m = _convert_chat_messages([prompt])[0]
        t = self.tokenizer.apply_chat_template(m, add_generation_prompt=True, return_dict=True)
        return len(t['input_ids'])

def  _convert_base_messages(inputs):
    outputs = []
    for _input in inputs:
        if isinstance(_input, str):
            outputs.append(_input)
        else:
            messages = []
            for item in _input:
                messages.append(item['prompt'])
            outputs.append(''.join(messages))
    return outputs


# class LLaDABaseModel(LLaDAwithChatTemplate):

#     def __init__(self,
#                  path: str,
#                  model_kwargs: dict = dict(),
#                  tokenizer_path: Optional[str] = None,
#                  tokenizer_kwargs: dict = dict(),
#                  peft_path: Optional[str] = None,
#                  peft_kwargs: dict = dict(),
#                  tokenizer_only: bool = False,
#                  generation_kwargs: dict = dict(),
#                  max_seq_len: Optional[int] = None,
#                  pad_token_id: Optional[int] = None,
#                  stop_words: Optional[str] = [],
#                  mask_token_ids: int = 126336,
#                  gen_method: str = "",
#                  remask_confidence: str = "probability",
#                  cfg_scale: float = 0.,
#                  add_gen_length: int = 4,
#                  add_gen_total_length: int = -1,
#                  gen_steps: int = 4,
#                  gen_temp_x0: float = 0.,
#                  gen_temp_remask: float = 0.,
#                  **other_kwargs):

#         self.logger = get_logger()
#         self.path = path
#         self.tokenizer_only = tokenizer_only
#         self.template_parser = LMTemplateParser()
#         self.max_seq_len = _get_possible_max_seq_len(max_seq_len, path)
#         self._load_tokenizer(tokenizer_path or path, tokenizer_kwargs, pad_token_id)
#         if not tokenizer_only:
#             self._load_model(path=path, kwargs=model_kwargs, peft_path=peft_path, peft_kwargs=peft_kwargs)
#         self.generation_kwargs = generation_kwargs
#         self.stop_words = stop_words

#         self.mask_token_ids = mask_token_ids
#         self.gen_method = gen_method
#         self.remask_confidence = remask_confidence
#         self.cfg_scale = cfg_scale
#         self.add_gen_length = add_gen_length
#         self.add_gen_total_length = add_gen_total_length
#         self.gen_steps = gen_steps
#         self.gen_temp_x0 = gen_temp_x0
#         self.gen_temp_remask = gen_temp_remask

#         for k, v in other_kwargs.items():
#             if v is not None:
#                 self.logger.warning(f'Unused argument {k}={v}')


#     def generate(self,
#                  inputs: List[str],
#                  max_out_len: int,
#                  min_out_len: Optional[int] = None,
#                  stopping_criteria: List[str] = [],
#                  **kwargs) -> List[str]:
#         messages = _convert_base_messages(inputs)
#         batch_size = len(messages)
#         assert batch_size == 1
#         print(messages)
#         print('--------------------------')

#         prompt = self.tokenizer(messages)["input_ids"]
#         prompt = torch.tensor(prompt, dtype=torch.long, device=self.model.device)
#         stopping_criteria = list(set(stopping_criteria + self.stop_words))
#         print(stopping_criteria)
#         print('^^^^^^^^^^^^^^^^^^^^^^^^^')

#         if self.gen_method == 's_ar':
#             # step-2: conduct model forward to generate output
#             gen_total_length = self.add_gen_total_length if self.add_gen_total_length > 0 else max_out_len
#             if gen_total_length % self.add_gen_length == 0:
#                 num_steps = gen_total_length // self.add_gen_length
#             else:
#                 num_steps = gen_total_length // self.add_gen_length + 1

#             seq = prompt.clone()
#             stop_generate = False
#             input_length = prompt.shape[1] + self.add_gen_total_length if self.add_gen_total_length > 0 else -1
#             for _ in range(num_steps):
#                 mask_seq = torch.full((seq.size(0), self.add_gen_length), self.mask_token_ids, device=self.model.device)
#                 seq = torch.cat([seq, mask_seq], dim=-1)
#                 prompt_index = torch.arange(seq.shape[1], device=self.model.device) < prompt.shape[1]

#                 transfer_tokens = self.get_transfer_tokens()
#                 for transfer_token in transfer_tokens:
#                     seq = self.gen_per_steps(seq, prompt_index, transfer_token, input_length)

#                 generated_answer = self.tokenizer.decode(seq[0][prompt.shape[1]:], skip_special_tokens=False)
#                 for stop_seq in stopping_criteria:
#                     if stop_seq in generated_answer:
#                         stop_generate = True
#                         generated_answer = generated_answer.split(stop_seq)[0]

#                 if stop_generate:
#                     break

#         elif self.gen_method == 'diff':
#             seq = torch.full((1, prompt.shape[1] + self.add_gen_length), self.mask_token_ids, device=self.model.device)
#             prompt_index = torch.arange(seq.shape[1], device=self.model.device) < prompt.shape[1]
#             seq[0, :prompt.shape[1]] = prompt[0]
            
#             transfer_tokens = self.get_transfer_tokens()
#             for transfer_token in transfer_tokens:
#                 seq = self.gen_per_steps(seq, prompt_index, transfer_token)

#             generated_answer = self.tokenizer.decode(seq[0][prompt.shape[1]:], skip_special_tokens=False)
#             for stop_seq in stopping_criteria:
#                 if stop_seq in generated_answer:
#                     generated_answer = generated_answer.split(stop_seq)[0]
#         else:
#             raise NotImplementedError(self.gen_method)

#         # remove special tokens
#         decodeds_ids = self.tokenizer(generated_answer)["input_ids"]
#         decodeds = self.tokenizer.decode(decodeds_ids, skip_special_tokens=True)
#         print(decodeds)
#         print('*****************************************************')
#         return [decodeds]

#     def get_ppl(self, inputs: List[str], mask_length: Optional[List[int]] = None) -> List[float]:
#         """Get perplexity scores given a list of inputs.

#         Args:
#             inputs (List[str]): A list of strings.
#             mask_length (Optional[List[int]]): A list of mask lengths. If
#                 provided, the perplexity scores will be calculated with the
#                 first mask_length[i] tokens masked out. It's okay to skip
#                 its implementation if advanced features in PPLInfernecer is
#                 not needed.

#         Returns:
#             List[float]: A list of perplexity scores.
#         """
#         assert self.tokenizer.pad_token
#         import torch
#         import torch.nn.functional as F
#         pad_token_id = self.tokenizer.pad_token_id
#         messages = _convert_base_messages(inputs)

#         tokenize_kwargs = dict(
#             return_tensors='pt',
#             padding=True,
#             truncation=True,
#             add_special_tokens=True,
#             max_length=self.max_seq_len
#         )
#         tokens = self.tokenizer.batch_encode_plus(messages, **tokenize_kwargs)
#         tokens = {k: v.to(self.model.device) for k, v in tokens.items()}
#         outputs = self.model(**tokens)[0]

#         batch_size, seq_len, vocab_size = outputs.shape
#         shift_logits = outputs[:, :-1, :].contiguous().float()
#         shift_labels = tokens['input_ids'][:, 1:].contiguous()
#         loss = F.cross_entropy(
#             shift_logits.view(-1, vocab_size),
#             shift_labels.view(-1),
#             ignore_index=pad_token_id,
#             reduction='none').view(batch_size, seq_len - 1)
#         lens = (tokens['input_ids'] != pad_token_id).sum(-1).cpu().numpy()

#         if mask_length is not None:
#             import numpy as np
#             mask = torch.zeros_like(shift_labels)  # [batch,seqlen]
#             for i in range(len(mask)):
#                 for j in range(mask_length[i] - 1, len(mask[i])):
#                     mask[i][j] = 1
#             loss = loss * mask
#             lens -= np.array(mask_length)

#         ce_loss = loss.float().sum(-1).cpu().detach().numpy() / lens
#         return ce_loss

#     def get_loglikelihood(self, inputs: List[str], conts:  List[str]) -> List[float]:
#         mask_length = [self.get_token_len(c, add_special_tokens=False) for c in conts]
#         return - self.get_ppl(inputs, mask_length)

#     def get_token_len(self, prompt: str, add_special_tokens: bool=True) -> int:
#         m = _convert_base_messages([prompt])[0]
#         t = self.tokenizer(m, add_special_tokens=add_special_tokens)
#         return len(t['input_ids'])