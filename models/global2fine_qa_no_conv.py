"""
The model without the convolutional block under the global-to-local architecture.
"""


#-*- coding: utf-8 -*-

import torch
import torch.nn as nn

import os
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
    squad_convert_examples_to_features,
)

from .outlook import *


class Global2Fine_NoConv(nn.Module):

    def __init__(self, model_name_or_path, config_name="", cache_dir="", embed_dims=768, num_layers=4, \
        downsamples=False, num_heads=1, out_kernel=[3,768], out_padding=1, out_stride=1, mlp_ratios=3, \
        qkv_bias=False, qk_scale=False, attn_drop_rate=0., norm_layer=nn.LayerNorm, outlook_attention=True):

        super().__init__()

        self.embed_dims = embed_dims

        self.backbone_name = model_name_or_path

        self.config = AutoConfig.from_pretrained(
                                    config_name if config_name else model_name_or_path,
                                    cache_dir=cache_dir if cache_dir else None,
                                    )

        self.backbone = AutoModel.from_pretrained(
                                    model_name_or_path,
                                    from_tf=bool(".ckpt" in model_name_or_path),
                                    config=self.config,
                                    cache_dir=cache_dir if cache_dir else None,
                                )

        self.outlookers = None

        self.outlooker_blocks = self.build_outlooker_blocks(Outlooker, embed_dims, num_layers,
                                         downsample=downsamples, num_heads=num_heads,
                                         kernel_size=out_kernel, stride=out_stride,
                                         padding=out_padding, mlp_ratio=mlp_ratios,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         attn_drop=attn_drop_rate, norm_layer=norm_layer)

        self.outlookers = nn.ModuleList(self.outlooker_blocks)

        self.qa_outputs = nn.Linear(in_features=embed_dims, out_features=2)

        self._keys_to_ignore_on_save = None

    def build_outlooker_blocks(self, block_fn, dim, num_layers, num_heads=1, kernel_size=3,
                 padding=1,stride=1, mlp_ratio=3., qkv_bias=False, qk_scale=None,
                 attn_drop=0, drop_path_rate=0., **kwargs):
        """
        generate outlooker layer in stage1
        return: outlooker layers
        """
        blocks = []
        for block_idx in range(num_layers):
            block_dpr = drop_path_rate * (block_idx +
                                          num_layers) / ((num_layers - 1)+1)
            '''
            blocks.append(block_fn(dim, kernel_size=kernel_size, padding=padding,
                                   stride=stride, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                   qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                                   drop_path=block_dpr))
            '''
            blocks.append(Outlooker(1,kernel_size=(3,self.embed_dims),stride=(1,self.embed_dims)))
            #blocks.append(Outlooker(dim=1, num_heads=1,kernel_size=3, padding=1))

        #blocks = nn.Sequential(*blocks)

        return blocks

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        save_config: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        **kwargs,
    ):
        """
        Save a model and its configuration file to a directory, so that it can be re-loaded using the
        `:func:`~transformers.PreTrainedModel.from_pretrained`` class method.

        Arguments:
            save_directory (:obj:`str` or :obj:`os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
            save_config (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to save the config of the model. Useful when in distributed training like TPUs and need
                to call this function on all processes. In this case, set :obj:`save_config=True` only on the main
                process to avoid race conditions.
            state_dict (nested dictionary of :obj:`torch.Tensor`):
                The state dictionary of the model to save. Will default to :obj:`self.state_dict()`, but can be used to
                only save parts of the model or if special precautions need to be taken when recovering the state
                dictionary of a model (like when using model parallelism).
            save_function (:obj:`Callable`):
                The function to use to save the state dictionary. Useful on distributed training like TPUs when one
                need to replace :obj:`torch.save` by another method.
            push_to_hub (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to push your model to the Hugging Face model hub after saving it.

                .. warning::

                    Using :obj:`push_to_hub=True` will synchronize the repository you are pushing to with
                    :obj:`save_directory`, which requires :obj:`save_directory` to be a local clone of the repo you are
                    pushing to if it's an existing folder. Pass along :obj:`temp_dir=True` to use a temporary directory
                    instead.

            kwargs:
                Additional key word arguments passed along to the
                :meth:`~transformers.file_utils.PushToHubMixin.push_to_hub` method.
        """

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo = self._create_or_get_repo(save_directory, **kwargs)

        os.makedirs(save_directory, exist_ok=True)

        # Only save the model itself if we are using distributed training
        model_to_save = unwrap_model(self)

        # Attach architecture to the config
        model_to_save.config.architectures = [model_to_save.__class__.__name__]

        # Save the config
        if save_config:
            model_to_save.config.save_pretrained(save_directory)

        # Save the model
        if state_dict is None:
            state_dict = model_to_save.state_dict()

        # Handle the case where some state_dict keys shouldn't be saved
        if self._keys_to_ignore_on_save is not None:
            state_dict = {k: v for k, v in state_dict.items() if k not in self._keys_to_ignore_on_save}

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
        save_function(state_dict, output_model_file)

        if push_to_hub:
            url = self._push_to_hub(repo, commit_message=commit_message)


    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
    ):
        

        outputs = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0].unsqueeze(-1)
        #outputs: last_hidden_state, pooler_output, (hidden_states), (attentions)

        for layer in self.outlookers:
            sequence_output = layer(sequence_output)

        logits = self.qa_outputs(sequence_output.squeeze(-1))
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)


        outputs = (start_logits, end_logits,) + outputs[2:]

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)
      
def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (:obj:`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model
