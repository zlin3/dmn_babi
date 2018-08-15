import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from models.utils import pad_collate, visualizeSample
from models.utils import DataLoader as DL
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class InputModule(nn.Module):
    def __init__(self, hidden_size):
        super(InputModule, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size)
        for name, param in self.gru.state_dict().items():
            if 'weight' in name: init.xavier_normal(param)
        self.dropout = nn.Dropout(0.1)

    def forward(self, contexts, embedding):
        batch_size, num_sentence, num_token = contexts.size()
        hidden = self.initHidden(batch_size)
        # embedded_contexts is of shape (batch_size, num_sentence, num_token, embedding_size)
        c = contexts.view(batch_size, -1)
        c.transpose_(0, 1)
        embedded_contexts = embedding(c)
        embedded_contexts = self.dropout(embedded_contexts)
        factsComb, hidden = self.gru(embedded_contexts, hidden)
        # facts is now of shape (num_sentence, batch_size, embedding_size)
        facts = factsComb[(num_token-1)::num_token, :, :]
        return facts

    def initHidden(self, batch_size):
        return Variable(torch.zeros(1, batch_size, self.hidden_size, device=device))
   
class QuestionModule(nn.Module):
    def __init__(self, hidden_size):
        super(QuestionModule, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size)
        for name, param in self.gru.state_dict().items():
            if 'weight' in name: init.xavier_normal(param)
        #self.dropout = nn.Dropout(0.1)

    def forward(self, questions, embedding):
        batch_size = len(questions)
        questionsT = questions.transpose(0, 1)
        hidden = self.initHidden(batch_size)
        embedded_questions = embedding(questionsT)
        #print(questions.shape, embedded_questions.shape)
        #embedded_questions = self.dropout(embedded_questions)
        ques, hiddens_final = self.gru(embedded_questions, hidden)
        return hiddens_final

    def initHidden(self, batch_size):
        return Variable(torch.zeros(1, batch_size, self.hidden_size, device=device))


class AnswerModule(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(AnswerModule, self).__init__()
        self.z = nn.Linear(2 * hidden_size, vocab_size)
        init.xavier_normal(self.z.state_dict()['weight'])
        self.dropout = nn.Dropout(0.1)

    def forward(self, M, questions):
        M = self.dropout(M)
        concat = torch.cat([M, questions], dim=2).squeeze(0)
        z = self.z(concat)
        return z

class AttentionGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionGRUCell, self).__init__()
        self.Wr = nn.Linear(input_size, hidden_size)
        init.xavier_normal(self.Wr.state_dict()['weight'])
        self.W = nn.Linear(input_size, hidden_size)
        init.xavier_normal(self.W.state_dict()['weight'])
        self.Ur = nn.Linear(hidden_size, hidden_size)
        init.xavier_normal(self.Ur.state_dict()['weight'])
        self.U = nn.Linear(hidden_size, hidden_size)
        init.xavier_normal(self.U.state_dict()['weight'])

    def forward(self, fact, hidden, gate):
        z = F.sigmoid(self.Wr(fact) + self.Ur(hidden))
        h_hat = F.tanh(self.W(fact) + self.U(hidden))
        gate = gate.unsqueeze(1)
        h = gate * h_hat + (1 - gate) * hidden
        return h

class AttentionGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionGRU, self).__init__()
        self.hidden_size = hidden_size
        self.cell = AttentionGRUCell(input_size, hidden_size)

    def forward(self, facts, gates):
        num_sentence, batch_size, embedding_size = facts.size()
        hidden = self.initHidden(batch_size)
        for sid in range(num_sentence):
            fact = facts[sid, :, :]
            gate = gates[sid, :]
            hidden = self.cell(fact, hidden, gate)
        return hidden

    def initHidden(self, batch_size):
        return Variable(torch.zeros(batch_size, self.hidden_size, device=device))

class MemoryModule(nn.Module):
    def __init__(self, hidden_size):
        super(MemoryModule, self).__init__()
        self.agru = AttentionGRU(hidden_size, hidden_size)
        self.W1 = nn.Linear(4 * hidden_size, hidden_size)
        init.xavier_normal(self.W1.state_dict()['weight'])
        self.W2 = nn.Linear(hidden_size, 1)
        init.xavier_normal(self.W2.state_dict()['weight'])
        # in the original paper, a GRU is used
        self.memory_weight = nn.Linear(3 * hidden_size, hidden_size)
        init.xavier_normal(self.memory_weight.state_dict()['weight'])

    def cal_attention(self, facts, questions, prevM):
        num_sentence, batch_size, embedding_size = facts.size()
        # May not need the following
        #questions = questions.expand_as(facts)
        #prevM = prevM.expand_as(facts)
        z = torch.cat([facts * questions, facts * prevM, torch.abs(facts - questions), torch.abs(facts - prevM)], dim=2)
        z = z.view(-1, 4 * embedding_size)
        gates = F.tanh(self.W1(z))
        gates = self.W2(gates)
        gates = gates.view(num_sentence, -1)
        gates = F.softmax(gates)
        return gates

    def forward(self, facts, questions, prevM):
        gates = self.cal_attention(facts, questions, prevM)
        episode = self.agru(facts, gates)
        concat = torch.cat([prevM.squeeze(0), episode, questions.squeeze(0)], dim=1)
        nextM = F.relu(self.memory_weight(concat))
        nextM = nextM.unsqueeze(0)
        return nextM


class DMN(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_hops=3):
        super(DMN, self).__init__()
        self.hidden_size = hidden_size
        self.num_hops = num_hops
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.input_module = InputModule(hidden_size)
        self.question_module = QuestionModule(hidden_size)
        self.memory_module = MemoryModule(hidden_size)
        self.answer_module = AnswerModule(vocab_size, hidden_size)
        self.criterion = nn.CrossEntropyLoss(size_average=False)

    def forward(self, contexts, questions):
        facts = self.input_module(contexts, self.embedding)
        qs = self.question_module(questions, self.embedding)
        memory = qs
        for hop in range(self.num_hops):
            memory = self.memory_module(facts, qs, memory)
        out = self.answer_module(memory, qs)
        return out

    def get_loss(self, contexts, questions, answers):
        out = self.forward(contexts, questions)
        loss = self.criterion(out, answers)
        reg_loss = 0
        for param in self.parameters():
            reg_loss += 0.001 * torch.sum(param * param)
        preds = F.softmax(out)
        _, pred_ids = torch.max(preds, dim=1)
        corrects = (pred_ids.data == answers.data)
        acc = torch.mean(corrects.float())
        return loss + reg_loss, acc, pred_ids

def train(dset, batch_size, epochs, dmn):

    early_stopping_cnt = 0
    early_stopping_flag = False
    best_acc = 0
    optim = torch.optim.Adam(dmn.parameters())


    for epoch in range(epochs):
        dset.set_mode('train')
        train_loader = DL(
            dset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate
        )

        dmn.train()
        if not early_stopping_flag:
            total_acc = 0
            cnt = 0
            for batch_idx, data in enumerate(train_loader):
                optim.zero_grad()
                contexts, questions, answers = data
                b_size = contexts.size()[0]
                contexts = Variable(contexts.long())
                questions = Variable(questions.long())
                answers = Variable(answers)

                loss, acc, _ = dmn.get_loss(contexts, questions, answers)
                loss.backward()
                total_acc += acc * b_size
                cnt += b_size

                if batch_idx % 20 == 0:
                    #print(f'[Task {task_id}, Epoch {epoch}] [Training] loss : {loss.data[0]: {10}.{8}}, acc : {total_acc / cnt: {5}.{4}}, batch_idx : {batch_idx}')
                    print('[Epoch %d] [Training] loss : %f, acc : %f, batch_idx : %d' % (epoch, loss.data[0], (total_acc/cnt), batch_idx))
                optim.step()

            dset.set_mode('valid')
            valid_loader = DL(
                dset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate
            )

            dmn.eval()
            total_acc = 0
            cnt = 0
            for batch_idx, data in enumerate(valid_loader):
                contexts, questions, answers = data
                b_size = contexts.size()[0]
                contexts = Variable(contexts.long())
                questions = Variable(questions.long())
                answers = Variable(answers)

                _, acc, _ = dmn.get_loss(contexts, questions, answers)
                total_acc += acc * batch_size
                cnt += batch_size

            total_acc = total_acc / cnt
            if total_acc > best_acc:
                best_acc = total_acc
                best_state = dmn.state_dict()
                early_stopping_cnt = 0
            else:
                early_stopping_cnt += 1
                if early_stopping_cnt > 20:
                    early_stopping_flag = True

            #print(f'[Run {run}, Task {task_id}, Epoch {epoch}] [Validate] Accuracy : {total_acc: {5}.{4}}')
            print('[Epoch %d] [Validate] Accuracy : %f' % (epoch, total_acc))
            if total_acc == 1.0:
                break
        else:
            #print(f'[Run {run}, Task {task_id}] Early Stopping at Epoch {epoch}, Valid Accuracy : {best_acc: {5}.{4}}')
            print('Early Stopping at Epoch %d, Valid Accuracy : %f' % (epoch, best_acc))
            break

    dmn.load_state_dict(best_state)

#    optim = torch.optim.Adam(dmn.parameters())
#    losses = []
#    accs = []
#    early_stopping_cnt = 0
#    early_stopping_flag = False
#    best_acc = 0
#
#    for e in range(epoch):
#        print('epoch %d is in progress' % e)
#        if not early_stopping_flag:
#            dset_train.set_mode('train')
#            train_loader = DL(dset_train, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
#            dmn.train()
#            total_acc = 0
#            cnt = 0
#            for batch_idx, data in enumerate(train_loader):
#                contexts, questions, answers = data
#                b_size = len(contexts)
#                optim.zero_grad()
#                c = contexts.long()
#                q = questions.long()
#                loss, acc, _ = dmn.get_loss(c, q, answers)
#                losses.append(loss.item())
#                accs.append(acc.item())
#                loss.backward()
#                optim.step()
#                total_acc += acc * b_size
#                cnt += b_size
#                #if batch_idx == 50: break
#            if e % 16 == 0:
#                plt.figure()
#                plt.plot(losses)
#                plt.title('training loss')
#                plt.show()
#                plt.figure()
#                plt.plot(accs)
#                plt.title('training accuracy')
#                plt.show()
#
#            dset_train.set_mode('valid')
#            valid_loader = DL(dset_train, batch_size=batch_size, shuffle=False, collate_fn=pad_collate)
#
#            dmn.eval()
#            total_acc = 0
#            cnt = 0
#            for batch_idx, data in enumerate(valid_loader):
#                contexts, questions, answers = data
#                b_size = len(contexts)
#                optim.zero_grad()
#                c = contexts.long()
#                q = questions.long()
#
#                _, acc, _ = dmn.get_loss(c, q, answers)
#                total_acc += acc * b_size
#                cnt += b_size
#
#            total_acc = total_acc / cnt
#            if total_acc > best_acc:
#                best_acc = total_acc
#                early_stopping_cnt = 0
#            else:
#                early_stopping_cnt += 1
#                if early_stopping_cnt > 20:
#                    early_stopping_flag = True
#
#            print('[Epoch %d] Validate Accuracy : %f' % (e, total_acc))
#            if total_acc == 1.0:
#                break

        
def test(dset_test, batch_size, dmn):
    test_loader = DL(dset_test, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
    vocab_size = len(dset_test.QA.VOCAB)
    losses = []
    accs = []
    for batch_idx, data in enumerate(test_loader):
        contexts, questions, answers = data
        b_size = len(contexts)
        c = Variable(contexts.long())
        q = Variable(questions.long())
        answers = Variable(answers)
        _, acc, topi = dmn.get_loss(c, q, answers)
        accs.append(acc.item())
        if batch_idx % 10 == 0:
            visualizeSample(dset_test, contexts[0], questions[0], answers[0], topi[0])
        #if batch_idx == 2: break
    accuracy = sum(accs) / len(accs)
    print("test accuracy is: %f" % accuracy)
