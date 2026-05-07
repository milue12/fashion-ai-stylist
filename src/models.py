# 파일명: models.py
import torch
import torch.nn as nn

class H1Regressor(nn.Module):
    def __init__(self, clip_model):
        super().__init__(); self.clip=clip_model.visual; D=self.clip.output_dim
        for p in self.clip.parameters(): p.requires_grad=False
        self.head=nn.Sequential(nn.LayerNorm(D),nn.Linear(D,D//2),nn.GELU(),nn.Linear(D//2,1))
    def forward(self,x):
        with torch.no_grad(): embedding=clip_model.visual(x)
        output=self.head(embedding).squeeze(1); return torch.sigmoid(output)*9.0+1.0

class SiameseModel(nn.Module):
    def __init__(self, clip_model):
        super().__init__(); self.encoder=clip_model.visual; D=self.encoder.output_dim; P=128
        for p in self.encoder.parameters(): p.requires_grad=False
        self.projection_head=nn.Sequential(nn.Linear(D,D),nn.ReLU(),nn.Linear(D,P))
    def forward(self,x):
        with torch.no_grad(): raw_emb=self.encoder(x)
        return self.projection_head(raw_emb)

class H3Classifier(nn.Module):
    def __init__(self, clip_model, num_classes=13):
        super().__init__(); self.encoder=clip_model.visual; D=self.encoder.output_dim
        for p in self.encoder.parameters(): p.requires_grad=False
        self.classifier_head=nn.Sequential(nn.LayerNorm(D),nn.Linear(D,D//2),nn.ReLU(),nn.Linear(D//2,num_classes))
    def forward(self,x):
        with torch.no_grad(): embedding=self.encoder(x)
        return self.classifier_head(embedding)

class H4Classifier(nn.Module):
    def __init__(self, clip_model):
        super().__init__(); self.encoder=clip_model.visual; D=self.encoder.output_dim
        for p in self.encoder.parameters(): p.requires_grad=False
        self.classifier_head=nn.Sequential(nn.LayerNorm(D),nn.Linear(D,D//4),nn.ReLU(),nn.Linear(D//4,1))
    def forward(self,x):
        with torch.no_grad(): embedding=self.encoder(x)
        return self.classifier_head(embedding).squeeze(-1)