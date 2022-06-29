import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from generate_data import generate_training_data


class ReplayBuffer:
  def __init__(self,max_size,input_shape,output_shape):
     self.mem_size=max_size
     self.mem_cntr=0
     # input x
     self.x=np.zeros((self.mem_size,*input_shape),dtype=np.float32)
     # output y
     self.y=np.zeros((self.mem_size,*output_shape),dtype=np.float32)

     self.filename='simul_data.buffer'

  def store_data(self,x,y):
     index=self.mem_cntr % self.mem_size
     self.x[index]=x
     self.y[index]=y
     self.mem_cntr +=1

  def resize(self,newsize):
     assert(newsize>self.mem_size)
     index=self.mem_cntr % self.mem_size
     input_shape=self.x.shape
     xnew=np.zeros((newsize,*input_shape[1:]),dtype=np.float32)
     xnew[:self.mem_size,:]=self.x
     output_shape=self.y.shape
     ynew=np.zeros((newsize,*output_shape[1:]),dtype=np.float32)
     ynew[:self.mem_size,:]=self.y
     self.x=xnew
     self.y=ynew
     self.mem_size=newsize

  def sample_minibatch(self,batch_size):
     filled=min(self.mem_cntr,self.mem_size)
     batch=np.random.choice(filled,batch_size,replace=False)
     x=self.x[batch]
     y=self.y[batch]

     return x,y

  def save_checkpoint(self,filename=None):
      if filename is None:
        with open(self.filename,'wb') as f:
          pickle.dump(self,f)
      else:
        with open(filename,'wb') as f:
          pickle.dump(self,f)

  def load_checkpoint(self,filename=None):
      if filename is None:
        with open(self.filename,'rb') as f:
          temp=pickle.load(f)
          self.mem_size=temp.mem_size
          self.mem_cntr=temp.mem_cntr
          self.x=temp.x
          self.y=temp.y
      else:
        with open(filename,'rb') as f:
          temp=pickle.load(f)
          self.mem_size=temp.mem_size
          self.mem_cntr=temp.mem_cntr
          self.x=temp.x
          self.y=temp.y


########################################################
# The following taken from 
# https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
def scaled_dot_product(q,k,v):
    d_k=q.size()[-1]
    attn_logits=torch.matmul(q,k.transpose(-2,-1))
    attn_logits /= math.sqrt(d_k)
    attention=F.softmax(attn_logits,dim=-1)
    values=torch.matmul(attention,v)
    return values, attention


class MultiheadAttention(nn.Module):
    def __init__(self,input_dim,embed_dim,num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be a multiple of number of heads"
        self.embed_dim=embed_dim
        self.num_heads=num_heads
        self.head_dim=embed_dim//num_heads

        # 3x below for stacking all q,k,v projections into one
        self.qkv_proj=nn.Linear(input_dim,3*embed_dim)
        self.o_proj=nn.Linear(embed_dim,embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self,x,return_attention=False):
        batch_size,embed_dim=x.size()
        qkv=self.qkv_proj(x)
        qkv=qkv.reshape(batch_size,self.num_heads,3*self.head_dim)
        q,k,v=qkv.chunk(3,dim=-1)

        values,attention=scaled_dot_product(q,k,v)
        values=values.reshape(batch_size,embed_dim)
        o=self.o_proj(values)

        if return_attention:
            return o,attention
        else:
            return o


class EncoderBlock(nn.Module):
    def __init__(self,input_dim,num_heads,dim_feedforward,dropout=0.0):
        # Inputs: input_dim - dimensions of input
        # num_heads : number of heads in the attention block
        # dim_feedforward : dimensions of the hidden layer
        # dropout : dropout
        super().__init__()
        self.self_attn=MultiheadAttention(input_dim,input_dim,num_heads)

        self.linear_net=nn.Sequential(
            nn.Linear(input_dim,dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward,input_dim)
        )

        self.norm1=nn.LayerNorm(input_dim)
        self.norm2=nn.LayerNorm(input_dim)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        attn_out=self.self_attn(x)
        x = x + self.dropout(attn_out)
        x=self.norm1(x)

        linear_out=self.linear_net(x)
        x = x + self.dropout(linear_out)
        x=self.norm2(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self,num_layers,input_dim,model_dim,num_classes,num_heads,dropout=0.0):
        super().__init__()
        self.input_net=nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(input_dim,model_dim)
        )
        self.layers=nn.ModuleList([EncoderBlock(input_dim=model_dim,dim_feedforward=model_dim,num_heads=num_heads,dropout=dropout) for _ in range(num_layers)])
        self.output_net=nn.Sequential(
            nn.Linear(model_dim,model_dim),
            nn.LayerNorm(model_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(model_dim,num_classes)
        )

    def forward(self,x):
        x=self.input_net(x)
        for l in self.layers:
            x=l(x)
        x=self.output_net(x)
        return torch.sigmoid(x)

    @torch.no_grad()
    def get_attention_maps(self,x):
        x=self.input_net(x)
        attention_maps=[]
        for l in self.layers:
            _, attn_map=l.self_attn(x,return_attention=True)
            attention_maps.append(attn_map)
            x=l(x)
        return attention_maps

########################################################
