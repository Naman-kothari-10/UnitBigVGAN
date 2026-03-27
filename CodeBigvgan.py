import torch
import torch.nn as nn
import torch.nn.functional as F
from bigvgan import BigVGAN

class CodeGenerator(BigVGAN):
    def __init__(self, h, use_cuda_kernel=False):
        if not isinstance(h, dict):
            h = vars(h)
        super().__init__(h, use_cuda_kernel=use_cuda_kernel)
            
        self.unit_lut = nn.Embedding(h["num_kmeans_units"]+1, h["unit_emb_dim"])# takes input uncollapsed units, + 1 for padding idx
        self.is_multilingual = h.get("is_multilingual", False)
        self.is_multispkr = h.get("is_multispkr", False)
        if self.is_multilingual:
            n_langs = h["n_langs"]
            lang_emb_dim = h["lang_emb_dim"]
            self.lang_lut = nn.Embedding(n_langs,lang_emb_dim)

    def forward(self, **kwargs):
        # print(f"Units shape: {kwargs['code'].shape}")
        x = self.unit_lut(kwargs["code"]).transpose(1,2)
        # print(f"Units emb shape: {x.shape}")
        T = x.shape[-1]
        
        if(self.is_multispkr):
            assert "spkr" in kwargs
            # print(f"Spk emb shape: {kwargs['spkr'].shape}")
            spk_emb = kwargs["spkr"].unsqueeze(-1)
            # print(f"Spk emb unsq shape: {spk_emb.shape}")
            spk_emb = spk_emb.expand(-1,-1,T)
            # print(f"Spk emb exp shape: {spk_emb.shape}")
            # import sys
            # sys.exit()
            
            x = torch.cat([spk_emb, x],dim=1)
            # print(f"Input to BigVGAN shape after spk emb cat: {x.shape}")
            
        if(self.is_multilingual):
            assert "lang_ids" in kwargs
            # print(f"Lang ids shape: {kwargs['lang_ids'].shape}") #torch.Size([16, 1])                                                                                                                              
            lang_emb = self.lang_lut(kwargs["lang_ids"])
            lang_emb = lang_emb.unsqueeze(-1)
            lang_emb = lang_emb.expand(-1, -1, T) 
            x = torch.cat([lang_emb,x],dim=1)
            
        out = super().forward(x)
        return out
            
class LIDClassifier(nn.Module): # As described in https://arxiv.org/pdf/2307.08655
    def __init__(self, in_channels, hidden_channels, num_languages):
        super().__init__()

        # First ConvLayer: Conv → ReLU → LayerNorm
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.ln1 = nn.LayerNorm(hidden_channels)

        # Second ConvLayer
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.ln2 = nn.LayerNorm(hidden_channels)

        # Final linear projection W
        self.proj = nn.Linear(hidden_channels, num_languages)

    def conv_layer(self, x, conv, ln):
    
        x = conv(x)                
        x = F.relu(x)
        x = x.transpose(1, 2)      
        x = ln(x)
        x = x.transpose(1, 2)      
        return x

    def forward(self, x):

        # Two stacked ConvLayers
        x = self.conv_layer(x, self.conv1, self.ln1)
        x = self.conv_layer(x, self.conv2, self.ln2)

        # Global average pooling over time
        x = x.mean(dim=-1)  
        # Linear projection + softmax
        logits = self.proj(x)  
        y_hat = F.softmax(logits, dim=-1)
        return y_hat, logits

                        
            
            
            
