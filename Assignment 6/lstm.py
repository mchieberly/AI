import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

torch.manual_seed(1)

lstm = nn.LSTM(3, 3)  # Input dim is 3, output dim is 3
inputs = [torch.randn(1, 3) for _ in range(5)]  # make a sequence of length 5

hidden = (torch.randn(1, 1, 3),
          torch.randn(1, 1, 3))
for i in inputs:
    out, hidden = lstm(i.view(1, 1, -1), hidden)

inputs = torch.cat(inputs).view(len(inputs), 1, -1)
hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # clean out hidden state
out, hidden = lstm(inputs, hidden)
#print(out)
#print(hidden)

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


training_data = [
    ("Clifford the big red dog ate the tasty chicken slowly".split(), ["NN", "DET", "ADJ", "ADJ", "NN", "V", "DET", "ADJ", "NN", "ADV"]),
    ("Everybody read that crazy book very quickly".split(), ["NN", "V", "DET", "ADJ", "NN", "ADV", "ADV"]), 
]
word_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:  # word has not been assigned an index yet
            word_to_ix[word] = len(word_to_ix)  # Assign each word with a unique index
#print(word_to_ix)
word_to_ix["chased"] = len(word_to_ix)
word_to_ix["The"] = len(word_to_ix)
tag_to_ix = {"DET": 0, "NN": 1, "V": 2, "ADJ": 3, "ADV": 4}  # Assign each tag with a unique index
ix_to_tag = ["DET", "NN", "V", "ADJ", "ADV"]

EMBEDDING_DIM = 6
HIDDEN_DIM = 6

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
    
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)
    #print(tag_scores)

for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in training_data:
        model.zero_grad()

        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        tag_scores = model(sentence_in)

        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

with torch.no_grad():
    def test(words, targets):

        for index, tag_score in enumerate(tag_scores):
            print(index, " ", words[index], "  ", ix_to_tag[np.argmax(tag_score).item()], "  (", targets[index], ")", sep = "")

    # inputs = prepare_sequence(training_data[0][0], word_to_ix)
    # tag_scores = model(inputs)
    # test(training_data[0][0], training_data[0][1])
    # print()

    # inputs = prepare_sequence(training_data[1][0], word_to_ix)
    # tag_scores = model(inputs)
    # test(training_data[1][0], training_data[1][1])

    # new_training_data1 = ("the crazy dog slowly ate the red book".split(), ["DET", "ADJ", "NN", "ADV", "V", "DET", "ADJ", "NN"])
    # inputs = prepare_sequence(new_training_data1[0], word_to_ix)
    # tag_scores = model(inputs)
    # test(new_training_data1[0], new_training_data1[1])
    # print()

    # new_training_data2 = ("Clifford the big chicken ate the dog quickly".split(), ["NN", "DET", "ADJ", "NN", "V", "DET", "NN", "ADV"])
    # inputs = prepare_sequence(new_training_data2[0], word_to_ix)
    # tag_scores = model(inputs)
    # test(new_training_data2[0], new_training_data2[1])
    # print()


    new_training_data3 = ("The crazy dog chased the big red chicken".split(), ["DET", "ADJ", "NN", "V", "DET", "ADJ", "ADJ", "NN"])
    inputs = prepare_sequence(new_training_data3[0], word_to_ix)
    tag_scores = model(inputs)
    test(new_training_data3[0], new_training_data3[1])