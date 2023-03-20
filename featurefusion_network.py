# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
TransT FeatureFusionNetwork class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from position_encoding import build_position_encoding


class FeatureFusionNetwork(nn.Module):

    def __init__(self, d_model=256, nhead=8, num_featurefusion_layers=4,
                 dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.device = torch.device('cuda')
        self.position_embedding = build_position_encoding(256)
        featurefusion_layer = FeatureFusionLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.encoder = Encoder(featurefusion_layer, num_featurefusion_layers)

        decoderCFA_layer = DecoderCFALayer(d_model, nhead, dim_feedforward, dropout, activation)
        decoderCFA_norm = nn.LayerNorm(d_model)
        self.decoder = Decoder(decoderCFA_layer, decoderCFA_norm)
        self.input_proj = nn.Conv2d(1024,d_model,kernel_size=1)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src_temp, src_search):
        # src_temp:[3,1024,h_temp,w_temp] src_search:[1,1024,h_src,w_src]

        M = src_temp.shape[0]
        src_temp = self.input_proj(src_temp)
        src_search_1 = self.input_proj(src_search)
        for i in range(0,M):
            if i == 0:
                src_search = src_search_1
            else:
                src_search = torch.cat((src_search,src_search_1),0)
        #print(src_temp.shape)
        #print(src_search.shape)

        h_temp, w_temp = src_temp.shape[2], src_temp.shape[3]
        mask_temp = torch.ones((M,h_temp,w_temp),dtype=torch.bool,device=self.device)
        mask_temp[:h_temp,:w_temp] = False
        #mask_temp1 = torch.ones((1,h_temp,w_temp),dtype=torch.bool,device=self.device)
        #mask_temp1[:h_temp,:w_temp] = False
        h_search, w_search = src_search.shape[2], src_search.shape[3]
        mask_search = torch.ones((M,h_search,w_search),dtype=torch.bool,device=self.device)
        mask_search[:h_search,:w_search] = False
        #mask_search1 = torch.ones((1,h_search,w_search),dtype=torch.bool,device=self.device)
        #mask_search1[:h_search,:w_search] = False
        #print(mask_temp.shape)
        #print(mask_search.shape)

        pos_temp = self.position_embedding(src_temp,mask_temp)
        pos_search = self.position_embedding(src_search,mask_search)
        #for i in range(0,M):
        #    if i == 0:
        #        pos_temp = pos_temp_1
        #        pos_search = pos_search_1
        #    else:
        #        pos_temp = torch.cat((pos_temp,pos_temp_1),0)
        #        pos_search = torch.cat((pos_search,pos_search_1),0)
        #print(pos_temp.shape)
        #print(pos_search.shape)

        src_temp = src_temp.flatten(2).permute(2, 0, 1)
        pos_temp = pos_temp.flatten(2).permute(2, 0, 1)
        src_search = src_search.flatten(2).permute(2, 0, 1)
        pos_search = pos_search.flatten(2).permute(2, 0, 1)
        mask_temp = mask_temp.flatten(1)
        mask_search = mask_search.flatten(1)
        #print(src_temp.shape)
        #print(src_search.shape)
        #print(mask_temp.shape)
        #print(mask_search.shape)
        #print(pos_temp.shape)
        #print(pos_search.shape)

        memory_temp, memory_search = self.encoder(src1=src_temp, src2=src_search, src1_key_padding_mask=mask_temp, src2_key_padding_mask=mask_search, pos_src1=pos_temp, pos_src2=pos_search)

        hs = self.decoder(memory_search, memory_temp, tgt_key_padding_mask=mask_search, memory_key_padding_mask=mask_temp, pos_enc=pos_temp, pos_dec=pos_search)
        
        #print(hs.shape)
        #print(hs.unsqueeze(0).transpose(1, 2).shape)
        return hs.unsqueeze(0).transpose(1, 2)


class Decoder(nn.Module):

    def __init__(self, decoderCFA_layer, norm=None):
        super().__init__()
        self.layers = _get_clones(decoderCFA_layer, 1)
        self.norm = norm

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos_enc: Optional[Tensor] = None,
                pos_dec: Optional[Tensor] = None):
        output = tgt

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos_enc=pos_enc, pos_dec=pos_dec)

        if self.norm is not None:
            output = self.norm(output)

        return output

class Encoder(nn.Module):

    def __init__(self, featurefusion_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(featurefusion_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src1, src2,
                src1_mask: Optional[Tensor] = None,
                src2_mask: Optional[Tensor] = None,
                src1_key_padding_mask: Optional[Tensor] = None,
                src2_key_padding_mask: Optional[Tensor] = None,
                pos_src1: Optional[Tensor] = None,
                pos_src2: Optional[Tensor] = None):
        output1 = src1
        output2 = src2

        for layer in self.layers:
            output1, output2 = layer(output1, output2, src1_mask=src1_mask,
                                     src2_mask=src2_mask,
                                     src1_key_padding_mask=src1_key_padding_mask,
                                     src2_key_padding_mask=src2_key_padding_mask,
                                     pos_src1=pos_src1, pos_src2=pos_src2)

        return output1, output2


class DecoderCFALayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
       
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos_enc: Optional[Tensor] = None,
                     pos_dec: Optional[Tensor] = None):

        tgt2 = self.multihead_attn(query=tgt+pos_dec,
                                   key=memory+pos_enc,
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt


    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos_enc: Optional[Tensor] = None,
                pos_dec: Optional[Tensor] = None):

        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos_enc, pos_dec)

class FeatureFusionLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        

        self.self_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Implementation of Feedforward model
        self.linear11 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear12 = nn.Linear(dim_feedforward, d_model)

        self.linear21 = nn.Linear(d_model, dim_feedforward)
        self.dropout2 = nn.Dropout(dropout)
        self.linear22 = nn.Linear(dim_feedforward, d_model)

        self.norm11 = nn.LayerNorm(d_model)
        self.norm12 = nn.LayerNorm(d_model)
        self.norm13 = nn.LayerNorm(d_model)
        self.norm21 = nn.LayerNorm(d_model)
        self.norm22 = nn.LayerNorm(d_model)
        self.norm23 = nn.LayerNorm(d_model)
        self.dropout11 = nn.Dropout(dropout)
        self.dropout12 = nn.Dropout(dropout)
        self.dropout13 = nn.Dropout(dropout)
        self.dropout21 = nn.Dropout(dropout)
        self.dropout22 = nn.Dropout(dropout)
        self.dropout23 = nn.Dropout(dropout)

        self.activation1 = _get_activation_fn(activation)
        self.activation2 = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src1, src2,
                     src1_mask: Optional[Tensor] = None,
                     src2_mask: Optional[Tensor] = None,
                     src1_key_padding_mask: Optional[Tensor] = None,
                     src2_key_padding_mask: Optional[Tensor] = None,
                     pos_src1: Optional[Tensor] = None,
                     pos_src2: Optional[Tensor] = None):
        #ECA
        original_size = src1.shape[0]
        src1 = src1.reshape(-1,1,256)
        pos_src1_cross = pos_src1.reshape(-1,1,256)
        mask_src1_cross = src1_key_padding_mask.reshape(1,-1)
        #print("test")
        #print(src1.shape)
        #print(pos_src1_cross.shape)
        #for i in range(0,M):
        #    if i == 0:
        #        new_pos_src1 = pos_src1
        #    else:
        #        new_pos_src1 = torch.cat((new_pos_src1,pos_src1),0)

        q1 = k1 = src1 #+ pos_src1_cross
        #print("test")
        #print(q1.shape)
        #print(src1.shape)
        #print(src1_key_padding_mask.shape)
        
        src12 = self.self_attn1(q1, k1, value=src1, attn_mask=src1_mask,
                                key_padding_mask=mask_src1_cross)[0]
        src1 = src1 + self.dropout11(src12)
        src1 = self.norm11(src1)
        src1 = src1.reshape(original_size,-1,256)

        q2 = k2 = src2 + pos_src2
        #print(src2_key_padding_mask.shape)
        #print(q2.shape)
        src22 = self.self_attn1(q2, k2, value=src2, attn_mask=src2_mask,
                                key_padding_mask=src2_key_padding_mask)[0]
        src2 = src2 + self.dropout21(src22)
        src2 = self.norm21(src2)

        #CFA
        #src1 = torch.split(scr1,h_temp*w_temp,1)
        #for i in range(0,M):
        src12 = self.multihead_attn1(query=src1+pos_src1,
                                   key=src2+pos_src2,
                                   value=src2)[0]
        src22 = self.multihead_attn2(query=src2+pos_src2,
                                   key=src1+pos_src1,
                                   value=src1)[0]

        src1 = src1 + self.dropout12(src12)
        src1 = self.norm12(src1)
        src12 = self.linear12(self.dropout1(self.activation1(self.linear11(src1))))
        src1 = src1 + self.dropout13(src12)
        src1 = self.norm13(src1)

        src2 = src2 + self.dropout22(src22)
        src2 = self.norm22(src2)
        src22 = self.linear22(self.dropout2(self.activation2(self.linear21(src2))))
        src2 = src2 + self.dropout23(src22)
        src2 = self.norm23(src2)
            #if i == 0:
            #    newsrc1 = src1[i]
            #    newsrc2 = src2
            #else:
            #    newsrc1 = torch.cat((newsrc1,src1[i]),1)
            #    newsrc2 = torch.cat((newsrc2,src2),1)

        return src1, src2

    def forward(self, src1, src2,
                src1_mask: Optional[Tensor] = None,
                src2_mask: Optional[Tensor] = None,
                src1_key_padding_mask: Optional[Tensor] = None,
                src2_key_padding_mask: Optional[Tensor] = None,
                pos_src1: Optional[Tensor] = None,
                pos_src2: Optional[Tensor] = None,):

        return self.forward_post(src1, src2, src1_mask, src2_mask, src1_key_padding_mask, src2_key_padding_mask, pos_src1, pos_src2)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_featurefusion_network(d_model,dropout,nhead,dim_feedforward,num_featurefusion_layers):
    return FeatureFusionNetwork(
        d_model=d_model,
        dropout=dropout,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        num_featurefusion_layers=num_featurefusion_layers
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
