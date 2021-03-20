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
    def __init__(self, embedding, hidden_dim1, hidden_dim2, width):
        super(ASAL_CLS, self).__init__()

        self._embedding = embedding

        self.fc1 = nn.Linear(opt.embed_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.out_fc = nn.Linear(hidden_dim2 * width, 2)
        self._init_weights()

    def forward(self, x):
#         logger.debug('ASAL_CLS: input size={}'.format(x.size()))
        output = self._embedding.embedded(x)
#         logger.debug('ASAL_CLS: output0 size={}'.format(output.size()))
        output = F.relu(self.fc1(output))
#         logger.debug('ASAL_CLS: output1 size={}'.format(output.size()))
        output = F.relu(self.fc2(output))
        output = torch.reshape(output, (output.size()[0], -1))
#         logger.debug('ASAL_CLS: output2 size={}'.format(output.size()))
        output = self.out_fc(output)
#         logger.debug('ASAL_CLS: output3 size={}'.format(output.size()))
        output = nn.functional.log_softmax(output, dim=1)

        return output

    def _init_weights(self):
        nn.init.normal_(self.fc1.weight, 0, 0.01)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, 0, 0.01)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.out_fc.weight, 0, 0.01)
        nn.init.constant_(self.out_fc.bias, 0)


def create_model(embedding, dim1=40, dim2=40, width=15):
    torch.manual_seed(opt.seed)
    model = ASAL_CLS(embedding, dim1, dim2, width)
    loss = nn.NLLLoss()
    # loss = nn.MSELoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(),
                                 lr=opt.lr_asal,
                                 momentum=0,
                                 weight_decay=0)
    logger.debug(str(model))
    logger.debug(str(loss))
    logger.debug(str(optimizer))
    return model, loss, optimizer

