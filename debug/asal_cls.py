#!/usr/bin/env python

"""Baseline for relative time embedding: learn regression model in terms of
relative time.
"""

__author__ = 'Anna Kukleva'
__date__ = 'September 2018'

import torch
import torch.nn as nn
from torch.nn import functional as F

from ute.utils.arg_pars import opt
from ute.utils.logging_setup import logger


class ASAL_CLS(nn.Module):
    def __init__(self, embedding, hidden_dim1, hidden_dim2):
        super(ASAL_CLS, self).__init__()

        self._embedding = embedding

        self.fc1 = nn.Linear(opt.embed_dim, opt.hidden_dim1)
        self.fc2 = nn.Linear(opt.hidden_dim1, opt.hidden_dim2)
        self.out_fc = nn.Linear(hidden_dim2, 2)
        self._init_weights()

    def forward(self, x):
        output = F.relu(self.fc1(x))
        output = F.relu(self.fc2(output))
        output = self.out_fc(output)
        output = nn.functional.log_softmax(output, dim=1)

        return output

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def create_model(K, dim1=40, dim2=40):
    torch.manual_seed(opt.seed)
    model = ASAL_CLS(K, dim1, dim2).to(opt.device)
    loss = nn.NLLLoss()
    # loss = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=opt.lr * 0.1,
                                 weight_decay=opt.weight_decay)
    logger.debug(str(model))
    logger.debug(str(loss))
    logger.debug(str(optimizer))
    return model, loss, optimizer

