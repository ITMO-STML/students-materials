import torch.nn as nn
import torch
from blocks.features import ExtractFbanks64
from blocks.resnet_optional_deq import ResNetModel
from blocks.pooling import StatPoolLayer
from blocks.segment import MaxoutSegmentLevel
from blocks.head import CurricularAAM 

class VerificationModel(nn.Module):
    def __init__(self, fe, frame_level, pooling, segment):
        super(VerificationModel, self).__init__()

        self.fe = fe
        self.frame_level = frame_level
        self.pooling = pooling
        self.segment_level = segment
        # self.head = head

    def forward(self, x, label=None):
        x = self.fe(x)
        x = self.frame_level(x.unsqueeze(0))
        x = self.pooling(x)
        x = self.segment_level(x)
        # if not eval:
        #     x = self.head(x)
        return x    
    
    def verify(self, enroll: torch.Tensor, test: torch.Tensor):
        with torch.no_grad():
            enroll_emb = self(enroll).view(1, -1)
            test_emb = self(test).view(1, -1)

            return torch.cosine_similarity(enroll_emb, test_emb).item()
        

          
