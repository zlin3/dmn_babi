import torch
import torch.nn as nn
from torch import optim
from models.utils import pad_collate, visualizeSample
from models.utils import DataLoader as DL
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BaselineGRU2(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(BaselineGRU2, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru1 = nn.GRU(hidden_size, hidden_size)
        self.gru2 = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(2 * hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, context, question, hidden1, hidden2):
        embedded_c = self.embedding(context)
        embedded_q = self.embedding(question)
        output_c = embedded_c
        output_c, hidden1 = self.gru1(output_c, hidden1)
        output_q = embedded_q
        output_q, hidden2 = self.gru2(output_q, hidden2)

        # concatenate the last output for both context and question
        concat = torch.cat([output_c[-1,:,:], output_q[-1,:,:]], dim=1)
        pred = self.softmax(self.out(concat))
        return pred

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device), torch.zeros(1, batch_size, self.hidden_size, device=device)
    
def train(dset_train, batch_size, epoch, baseGRU):
    criterion = nn.NLLLoss()
    optim = torch.optim.Adam(baseGRU.parameters())
    losses = []
    accs = []
    for e in range(epoch):
        train_loader = DL(dset_train, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
        print('epoch %d is in progress' % e)
        for batch_idx, data in enumerate(train_loader):
            contexts, questions, answers = data
            b_size = len(contexts)
            optim.zero_grad()
            c = contexts.view(b_size, -1).long()
            q = questions.view(b_size, -1).long()
            c.transpose_(0, 1)
            q.transpose_(0, 1)
            hidden1, hidden2 = baseGRU.initHidden(b_size)
            out = baseGRU(c, q, hidden1, hidden2)
            topv, topi = out.data.topk(1)
            topi = topi.view(1, -1)
            acc = torch.mean((topi.data == answers.data).float())
            loss = criterion(out, answers)
            losses.append(loss.item())
            accs.append(acc.item())
            loss.backward()
            optim.step()
            #if batch_idx == 50: break
        if e % 16 == 0:
            plt.figure()
            plt.plot(losses)
            plt.title('training loss')
            plt.show()
            plt.figure()
            plt.plot(accs)
            plt.title('training accuracy')
            plt.show()
        
def test(dset_test, batch_size, baseGRU):
    test_loader = DL(dset_test, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
    vocab_size = len(dset_test.QA.VOCAB)
    losses = []
    accs = []
    for batch_idx, data in enumerate(test_loader):
        contexts, questions, answers = data
        b_size = len(contexts)
        c = contexts.view(b_size, -1).long()
        q = questions.view(b_size, -1).long()
        c.transpose_(0, 1)
        q.transpose_(0, 1)
        hidden1, hidden2 = baseGRU.initHidden(b_size)
        out = baseGRU(c, q, hidden1, hidden2)
        topv, topi = out.data.topk(1)
        topi = topi.view(1, -1).squeeze(0)
        acc = torch.mean((topi.data == answers.data).float())
        accs.append(acc.item())
        if batch_idx % 10 == 0:
            visualizeSample(dset_test, contexts[0], questions[0], answers[0], topi[0])
        #if batch_idx == 2: break
    accuracy = sum(accs) / len(accs)
    print("test accuracy is: %f" % accuracy)
