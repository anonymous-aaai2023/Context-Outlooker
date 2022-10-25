
#-*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

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

from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)

from .outlook_conv import *


class Global2Fine_Token_Classification(nn.Module):


    """
    First adopt Bert LM to encode the global representations (get the features before performeding the QA task).
    Then input the obtained global representations into Outlooker to further encode the local information.
    Finally, execute the QA task.
    """

    def __init__(self, model_name_or_path, config_name="", num_labels=9, cache_dir="", embed_dims=768, seq_len = 384, num_layers=4, \
        downsamples=False, num_heads=1, out_kernel=(3,300), out_padding=1, out_stride=(1,300), mlp_ratios=3, \
        qkv_bias=False, qk_scale=False, attn_drop_rate=0., norm_layer=nn.LayerNorm, outlook_attention=True, \
        n_filters=100, filter_sizes=[3,4,5]):

        super().__init__()

        self.out_dim = n_filters*len(filter_sizes)

        self.config = AutoConfig.from_pretrained(
                                    config_name if config_name else model_name_or_path,
                                    cache_dir=cache_dir if cache_dir else None,
                                    num_labels=num_labels,
                                    )

        self.backbone = AutoModel.from_pretrained(
                                    model_name_or_path,
                                    from_tf=bool(".ckpt" in model_name_or_path),
                                    config=self.config,
                                    cache_dir=cache_dir if cache_dir else None,
                                )

        self.num_labels = self.config.num_labels

        self.convs = nn.ModuleList([
                    nn.Conv2d(in_channels = 1,
                              out_channels = n_filters,
                              kernel_size = (fs, embed_dims+4),
                              padding = 2)
                    for fs in filter_sizes
                    ])

        self.outlookers = None

        self.outlooker_blocks = self.build_outlooker_blocks(Outlooker, embed_dims, num_layers,
                                         downsample=downsamples, num_heads=num_heads,
                                         kernel_size=out_kernel, stride=out_stride,
                                         padding=out_padding, mlp_ratio=mlp_ratios,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         attn_drop=attn_drop_rate, norm_layer=norm_layer)

        self.outlookers = nn.ModuleList(self.outlooker_blocks)
        self.reduce = nn.Linear(embed_dims,self.out_dim)

        #self.base_model_prefix ="bert"

        #self.bert_pool = BertPooler(self.out_dim)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.out_dim, self.num_labels)

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
            blocks.append(Outlooker(1))
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
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
        )
        
        seq_len = outputs[0].size()[1]

        #outputs[0]: [B,384,768]
        #global_output = self.reduce(outputs[0])

        #sequence_output = outputs[-1][5].unsqueeze(1)
        sequence_output = outputs[0].unsqueeze(1)
        #outputs: last_hidden_state, pooler_output, (hidden_states), (attentions)
        
        conved = [F.adaptive_max_pool1d(F.relu(conv(sequence_output)).squeeze(3),seq_len) for conv in self.convs]
        conved = torch.cat(conved,1).permute(0,2,1).unsqueeze(-1)

        for layer in self.outlookers:
            conved = layer(conved)

        #output_layer_input = global_output+conved.squeeze(-1)
        output_layer_input = conved.squeeze(-1)

        output_layer_input = self.dropout(output_layer_input)
        logits = self.classifier(output_layer_input)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class BertPooler(nn.Module):
    def __init__(self,hidden_num):
        super().__init__()
        self.dense = nn.Linear(hidden_num,hidden_num)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

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
