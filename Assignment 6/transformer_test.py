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
from torchtext.vocab import Vocab

from transformer import TransformerModel, emsize, nhead, d_hid, nlayers, dropout
from transformer import generate_square_subsequent_mask, get_batch, evaluate
import transformer_train as train
from transformer_train import ntokens, device

test_data = train.test_data
bptt = 35
criterion = nn.CrossEntropyLoss()


train_iter = WikiText2(split='train')
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])
ntokens = len(vocab)  # size of vocabulary

model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)

best_model_params_path = os.path.join('.', "best_model_params.pt")
model.load_state_dict(torch.load(best_model_params_path)) # load best model states
test_loss = evaluate(model, test_data, bptt, ntokens, criterion)

test_ppl = math.exp(test_loss)
print('=' * 89)
print(f'| End of testing | test loss {test_loss:5.2f} | '
      f'test ppl {test_ppl:8.2f}')
print('=' * 89)

v = Vocab(train.vocab)

def report(
      model,
      eval_data,
      bptt,
      ntokens,
      criterion,
      device='cuda'):

      model.eval()  # turn on evaluation mode
      total_loss = 0.
      src_mask = generate_square_subsequent_mask(bptt).to(device)
      with torch.no_grad():
            for i in range(0, eval_data.size(0) - 1, bptt):
                  data, targets = get_batch(eval_data, i, bptt)
                  seq_len = data.size(0)
                  if seq_len != bptt:
                        src_mask = src_mask[:seq_len, :seq_len]
            output = model(data, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += seq_len * criterion(output_flat, targets).item()
            for i in data:
                  for k in i:
                        print(v.lookup_token(int(k)), end = ' ')
                  print()
            iteration = 1
            for j in targets:
                  print(v.lookup_token(int(k)), end = ' ')
                  if iteration % 10 == 0:
                       print()
                  iteration += 1
      return total_loss / (len(eval_data) - 1)

test_loss = report(model, test_data, bptt, ntokens, criterion)