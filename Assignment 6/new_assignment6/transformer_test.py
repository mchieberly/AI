#!/usr/bin/python3

import math
import os
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from transformer import TransformerModel, emsize, nhead, d_hid, nlayers, dropout, evaluate

train_iter = WikiText2(split='train')
tokenizer = get_tokenizer('basic_english')
print('Building vocabulary ...')
vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])
ntokens = len(vocab)  # size of vocabulary

model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout)

best_model_params_path = os.path.join('.', "best_model_params.pt")
model.load_state_dict(torch.load(best_model_params_path)) # load best model states
test_loss = evaluate(model, test_data)

'''
test_ppl = math.exp(test_loss)
print('=' * 89)
print(f'| End of training | test loss {test_loss:5.2f} | '
      f'test ppl {test_ppl:8.2f}')
print('=' * 89)
'''
