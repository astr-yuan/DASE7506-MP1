"""GPT with rotary position encoding; all temporary tensors are call-local."""
import torch
from torch import nn
from torch.nn import functional as F
from model import GPT, Block


class RotaryBlock(Block):
    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(batch, length, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4)
        dim = width // self.heads
        frequency = 10000.0 ** (-torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim)
        angle = torch.arange(length, device=x.device, dtype=torch.float32)[:, None] * frequency[None, :]
        cos, sin = angle.cos().to(q.dtype), angle.sin().to(q.dtype)
        def rotate(z):
            even, odd = z[..., 0::2], z[..., 1::2]
            return torch.stack((even*cos-odd*sin, even*sin+odd*cos), dim=-1).flatten(-2)
        attended = F.scaled_dot_product_attention(rotate(q), rotate(k), v, is_causal=True)
        x = x + self.proj(attended.transpose(1, 2).reshape(batch, length, width))
        return x + self.mlp(self.norm2(x))


class RotaryGPT(GPT):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = dict(config)
        self.context = config['context']
        width = config['width']
        if width % config['heads'] or (width // config['heads']) % 2:
            raise ValueError('RoPE requires an even head dimension.')
        self.token = nn.Embedding(config['vocab'], width)
        self.blocks = nn.ModuleList([RotaryBlock(width, config['heads']) for _ in range(config['depth'])])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, config['vocab'], bias=False)
        self.apply(self.initialize)
        self.head.weight = self.token.weight

    def features(self, ids):
        if ids.shape[1] > self.context:
            raise ValueError('Context exceeds 256.')
        x = self.token(ids)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)



class RepetitionGPT(RotaryGPT):
    def __init__(self,config):
        super().__init__(config)
        self.copy_alpha=float(config.get('copy_alpha',0.0))
        self.copy_order=int(config.get('copy_order',2))
        self.copy_smoothing=float(config.get('copy_smoothing',2.0))
        if not 0<=self.copy_alpha<1 or not 1<=self.copy_order<=4 or self.copy_smoothing<=0:
            raise ValueError('Invalid causal-copy configuration.')

    def copy_distribution(self,ids):
        batch,length=ids.shape
        positions=torch.arange(length,device=ids.device)
        # Column j predicts its observed following token ids[j+1]. j<t makes
        # that token part of the current prefix, never the future target.
        matches=(ids[:,:,None]==ids[:,None,:]) & (positions[None,:]<positions[:,None])[None,:,:]
        chosen=matches
        for offset in range(1,self.copy_order):
            previous=ids.roll(offset,dims=1)
            valid=(positions>=offset)
            matches=matches & (previous[:,:,None]==previous[:,None,:]) & valid[None,:,None] & valid[None,None,:]
            chosen=torch.where(matches.any(-1,keepdim=True),matches,chosen)
        counts=chosen.sum(-1,keepdim=True).float()
        continuation=torch.cat((ids[:,1:],torch.zeros_like(ids[:,:1])),dim=1)
        histogram=torch.zeros(batch,length,self.config['vocab'],device=ids.device,dtype=torch.float32)
        histogram.scatter_add_(-1,continuation[:,None,:].expand(batch,length,length),chosen.float())
        return histogram/counts.clamp_min(1),counts

    def predict_log_probs(self,ids):
        logp=F.log_softmax(self(ids).float(),dim=-1)
        if self.copy_alpha==0:
            return logp
        copy,counts=self.copy_distribution(ids)
        weight=self.copy_alpha*counts/(counts+self.copy_smoothing)
        # The neural component stays strictly positive and preserves finite logs.
        return torch.logaddexp(logp+torch.log1p(-weight),torch.log((copy*weight).clamp_min(1e-30)))



def build_model(config):
    return RepetitionGPT(config) if config.get("rope", True) else GPT(config)
