from typing import Optional, Tuple, Union
import pdb
import sys
import copy
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention, GPTNeoXRotaryEmbedding, GPTNeoXLinearScalingRotaryEmbedding, GPTNeoXDynamicNTKScalingRotaryEmbedding, apply_rotary_pos_emb

class GPTNeoXAttention_OutScore(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "The hidden size is not divisble by the number of attention heads! Make sure to update them"
            )
        self.head_size = self.hidden_size // self.num_attention_heads
        self.rotary_ndims = int(self.head_size * config.rotary_pct)
        self._init_bias(config.max_position_embeddings)

        self.register_buffer("masked_bias", torch.tensor(-1e9), persistent=False)
        self._init_rope()

        self.norm_factor = self.head_size**-0.5
        self.query_key_value = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.attention_mask = None

    def _init_bias(self, max_positions, device=None):
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        if device is not None:
            self.bias = self.bias.to(device)

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = GPTNeoXRotaryEmbedding(
                self.rotary_ndims, self.config.max_position_embeddings, base=self.config.rotary_emb_base
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = GPTNeoXLinearScalingRotaryEmbedding(
                    self.rotary_ndims,
                    self.config.max_position_embeddings,
                    base=self.config.rotary_emb_base,
                    scaling_factor=scaling_factor,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = GPTNeoXDynamicNTKScalingRotaryEmbedding(
                    self.rotary_ndims,
                    self.config.max_position_embeddings,
                    base=self.config.rotary_emb_base,
                    scaling_factor=scaling_factor,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        position_ids: torch.LongTensor,
        head_mask: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ):
        has_layer_past = layer_past is not None

        # Compute QKV
        # Attention heads [batch, seq_len, hidden_size]
        #   --> [batch, seq_len, (np * 3 * head_size)]
        qkv = self.query_key_value(hidden_states)

        # [batch, seq_len, (num_heads * 3 * head_size)]
        #   --> [batch, seq_len, num_heads, 3 * head_size]
        new_qkv_shape = qkv.size()[:-1] + (self.num_attention_heads, 3 * self.head_size)
        qkv = qkv.view(*new_qkv_shape)

        # [batch, seq_len, num_attention_heads, 3 * head_size] --> 3 [batch, num_attention_heads, seq_len, head_size]
        query = qkv[..., : self.head_size].permute(0, 2, 1, 3)
        key = qkv[..., self.head_size : 2 * self.head_size].permute(0, 2, 1, 3)
        value = qkv[..., 2 * self.head_size :].permute(0, 2, 1, 3)

        # Compute rotary embeddings on rotary_ndims
        query_rot = query[..., : self.rotary_ndims]
        query_pass = query[..., self.rotary_ndims :]
        key_rot = key[..., : self.rotary_ndims]
        key_pass = key[..., self.rotary_ndims :]

        # Compute token offset for rotary embeddings (when decoding)
        seq_len = key.shape[-2]
        if has_layer_past:
            seq_len += layer_past[0].shape[-2]
        cos, sin = self.rotary_emb(value, seq_len=seq_len)
        query, key = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)
        query = torch.cat((query, query_pass), dim=-1)
        key = torch.cat((key, key_pass), dim=-1)

        # Cache QKV values
        if has_layer_past:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        present = (key, value) if use_cache else None

        # Compute attention
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        # Reshape outputs
        attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_size)
        attn_output = self.dense(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs

    @classmethod
    def _split_heads(cls, tensor, num_attention_heads, attn_head_size):
        """
        Splits hidden dim into attn_head_size and num_attention_heads
        """
        # tensor: [bs, seq_len, hidden_size]
        new_shape = tensor.size()[:-1] + (num_attention_heads, attn_head_size)
        # -> [bs, seq_len, num_attention_heads, attn_head_size]
        tensor = tensor.view(new_shape)
        # -> [bs, num_attention_heads, seq_len, attn_head_size]
        tensor = tensor.permute(0, 2, 1, 3)
        return tensor

    @classmethod
    def _merge_heads(cls, tensor, num_attention_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden dim
        """
        # tensor [bs, num_attention_heads, seq_len, attn_head_size]
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        # -> [bs, seq_len, num_attention_heads, attn_head_size]
        tensor = tensor.view(tensor.size(0), tensor.size(1), num_attention_heads * attn_head_size)
        # -> [bs, seq_len, hidden_size]
        return tensor

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        # q, k, v: [bs, num_attention_heads, seq_len, attn_head_size]
        # compute causal mask from causal mask buffer
        batch_size, num_attention_heads, query_length, attn_head_size = query.size()
        key_length = key.size(-2)

        # dynamically increase the causal mask with the key length, if needed.
        if key_length > self.bias.shape[-1]:
            self._init_bias(key_length, device=key.device)
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]

        query = query.view(batch_size * num_attention_heads, query_length, attn_head_size)
        key = key.view(batch_size * num_attention_heads, key_length, attn_head_size)
        attn_scores = torch.zeros(
            batch_size * num_attention_heads,
            query_length,
            key_length,
            dtype=query.dtype,
            device=key.device,
        )
        attn_scores = torch.baddbmm(
            attn_scores,
            query,
            key.transpose(1, 2),
            beta=1.0,
            alpha=self.norm_factor,
        )
        attn_scores = attn_scores.view(batch_size, num_attention_heads, query_length, key_length)

        mask_value = torch.finfo(attn_scores.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.tensor(mask_value, dtype=attn_scores.dtype).to(attn_scores.device)
        attn_scores = torch.where(causal_mask, attn_scores, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_scores = attn_scores + attention_mask

        attn_weights = nn.functional.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights.to(value.dtype)

        self.attention_mask = attn_weights.clone().detach()

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_weights = self.attention_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights

def convert_gptneox_output(model, config):

    for name, module in reversed(model._modules.items()):

        if len(list(module.children())) > 0:
            model._modules[name] = convert_gptneox_output(module, config)

        if isinstance(module, GPTNeoXAttention):
            model._modules[name] = GPTNeoXAttention_OutScore(config)
    return model

global activation 
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name+'input'] = input[0].detach()
        activation[name+'output'] = output.detach()
    return hook

if sys.argv[1] == '70m':
    config = AutoConfig.from_pretrained('EleutherAI/pythia-70m')
    model = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-70m')
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-70m')
elif sys.argv[1] == '160m':
    config = AutoConfig.from_pretrained('EleutherAI/pythia-160m')
    model = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-160m')
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-160m')
else:
    assert False

if sys.argv[2] == 'a':
    sentence1 = 'Every morning, as the city slowly awakens with the distant hum of traffic and the chirping of sparrows, John takes a moment to savor the peaceful ambiance before he walks his dog, Max, around the block, greeting familiar faces and enjoying the fresh air.'
    sentence2 = 'In the realm of physics, when water is subjected to a temperature of 100°C at one atmosphere of pressure, it undergoes a phase transition from liquid to gas, producing steam that has long been harnessed for various technological and culinary applications'
elif sys.argv[2] == 'b':
    sentence1 = 'Cats are known for their independent nature. Many people appreciate them for their low-maintenance lifestyle, often content with just a comfortable spot to nap and an occasional playtime'
    sentence2 = 'Rainforests are vital for the Earth\'s ecosystem. They provide a habitat for countless species, many of which are not found anywhere else. Additionally, they play a crucial role in regulating global climate and producing oxygen'
elif sys.argv[2] == 'c':
    sentence1 = 'The Eiffel Tower, an iconic landmark in Paris, was originally constructed as a temporary exhibit for the 1889 World\'s Fair. Over the years, it has become a symbol of the city\'s romance and architectural prowess, attracting millions of tourists annually.'
    sentence2 = 'The human digestive system is a complex network of organs working together to break down food into essential nutrients. Beginning with the mouth and ending at the small intestine, each part plays a crucial role in ensuring our bodies receive the energy and vitamins needed for daily function.'
elif sys.argv[2] == 'd':
    sentence1 = 'The Sahara Desert, stretching across North Africa, is the third largest desert in the world and is renowned for its vast sand dunes and scorching temperatures. Despite its harsh conditions, it\'s home to various unique species that have adapted to its extreme environment.'
    sentence2 = 'Novels, beyond their entertainment value, serve as mirrors to society, often reflecting cultural, social, and political nuances of their time. Authors like George Orwell and Jane Austen used their works to critique and provide insights into the world they lived in.'
elif sys.argv[2] == 'e':
    sentence1 = 'The Great Barrier Reef, located off the coast of Queensland in northeastern Australia, is the world\'s largest coral reef system. It boasts a staggering diversity of marine life and has been designated a World Heritage Site due to its ecological importance.'
    sentence2 = 'Mozart, a prodigious composer of the classical era, began composing at a very young age. His contributions to symphonic, chamber, operatic, and choral music are revered, and his compositions are celebrated for their technical command and emotional depth.'
elif sys.argv[2] == 'f':
    sentence1 = 'Mount Everest, rising majestically in the Himalayas, holds the title as the highest point on Earth\'s surface. Each year, adventurers and climbers from around the world embark on perilous journeys to conquer its peak, driven by ambition and the allure of its challenge.'
    sentence2 = 'Chocolate, with its rich history dating back to ancient Mesoamerican civilizations, has evolved from a ceremonial drink to a beloved global treat. Its production, from cocoa bean harvesting to the final product, impacts economies and cultures around the world.'
else:
    assert False

inputs1 = tokenizer.encode(sentence1, return_tensors="pt")
inputs2 = tokenizer.encode(sentence2, return_tensors="pt")

checkpoint = copy.deepcopy(model.state_dict())
model = convert_gptneox_output(model, config)
model.load_state_dict(checkpoint)

num_layers = len(model.gpt_neox.layers)
for layer_idx in range(num_layers):
    model.gpt_neox.layers[layer_idx].mlp.dense_h_to_4h.register_forward_hook(get_activation('layer_{}_mlp_fc1'.format(layer_idx)))

# sentence 1
activation = {}
attention_mask = {}
output = model(inputs1)
for name, m in model.named_modules():
    if isinstance(m, GPTNeoXAttention_OutScore):
        attention_mask[name] = m.attention_mask

embeddings_1 = copy.deepcopy(activation['layer_0_mlp_fc1output'].relu())
attention_1 = copy.deepcopy(attention_mask['gpt_neox.layers.0.attention'])

# sentence 2
activation = {}
attention_mask = {}
output = model(inputs2)
for name, m in model.named_modules():
    if isinstance(m, GPTNeoXAttention_OutScore):
        attention_mask[name] = m.attention_mask

embeddings_2 = copy.deepcopy(activation['layer_0_mlp_fc1output'].relu())
attention_2 = copy.deepcopy(attention_mask['gpt_neox.layers.0.attention'])

# Heavy-Hitter
activated_score_1 = embeddings_1.sum(1)
activated_score_2 = embeddings_2.sum(1)
activation_value_neuron = activated_score_1 + activated_score_2
activation_value_neuron = activation_value_neuron.reshape(-1)
hh_rank = activation_value_neuron.argsort() # small to large
for idx in range(1, 6): # analysis the top-10 neurons
    print('########### Analyze Top-{} Heavy-Hitters ###########'.format(idx))
    neuron_idx = hh_rank[-idx]
    activation_score_1_idx = activated_score_1[0, neuron_idx]
    activation_score_2_idx = activated_score_2[0, neuron_idx]
    # print('neuron idx {}, activation value for seq 1: {:.4f}, activation value for seq 2: {:.4f}'.format(neuron_idx, activation_score_1_idx, activation_score_2_idx))
    # Most activated tokens
    token_idx_1 = embeddings_1[0,:,neuron_idx].argmax()
    token_idx_2 = embeddings_2[0,:,neuron_idx].argmax()
    # print('Token idx-1 {}, Token idx-2 {}'.format(token_idx_1, token_idx_2))
    attn_map_for_token_1 = attention_1[0, :, token_idx_1, :].mean(0)
    attn_map_for_token_2 = attention_2[0, :, token_idx_2, :].mean(0)
    num_tokens_1 = attn_map_for_token_1.shape[0]
    num_tokens_2 = attn_map_for_token_2.shape[0]
    thres_1 = 2 / num_tokens_1
    thres_2 = 2 / num_tokens_2
    highlight_token_1 = torch.nonzero(attn_map_for_token_1 > thres_1).reshape(-1)
    highlight_token_2 = torch.nonzero(attn_map_for_token_2 > thres_2).reshape(-1)
    selected_tokens_1 = torch.tensor([inputs1[0,item] for item in highlight_token_1])
    selected_tokens_2 = torch.tensor([inputs2[0,item] for item in highlight_token_2])
    highlight_seq_1 = tokenizer.decode(selected_tokens_1)
    highlight_seq_2 = tokenizer.decode(selected_tokens_2)
    print('## Attention words 1: {}'.format(highlight_seq_1))
    print('## Attention words 2: {}'.format(highlight_seq_2))


