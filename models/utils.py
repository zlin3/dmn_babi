from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import numpy as np

def pad_collate(batch):
    max_context_sen_len = float('-inf')
    max_context_len = float('-inf')
    max_question_len = float('-inf')
    for elem in batch:
        context, question, _ = elem
        max_context_len = max_context_len if max_context_len > len(context) else len(context)
        max_question_len = max_question_len if max_question_len > len(question) else len(question)
        for sen in context:
            max_context_sen_len = max_context_sen_len if max_context_sen_len > len(sen) else len(sen)
    max_context_len = min(max_context_len, 70)
    for i, elem in enumerate(batch):
        _context, question, answer = elem
        _context = _context[-max_context_len:]
        context = np.zeros((max_context_len, max_context_sen_len))
        for j, sen in enumerate(_context):
            context[j] = np.pad(sen, (0, max_context_sen_len - len(sen)), 'constant', constant_values=0)
        question = np.pad(question, (0, max_question_len - len(question)), 'constant', constant_values=0)
        batch[i] = (context, question, answer)
    return default_collate(batch)

def visualizeSample(dset, context, question, answer, prediction):
    ivocab = dset.QA.IVOCAB
    for i, sentence in enumerate(context):
        s = ' '.join([ivocab[elem.item()] for elem in sentence])
        print('context %d: %s' % (i, s))
    q = ' '.join([ivocab[elem.item()] for elem in question])
    a = ' '.join([ivocab[answer.item()]])
    p = ' '.join([ivocab[prediction.item()]])
    print('question: %s' % q)
    print('answer: %s' % a)
    print('prediction: %s' % p)

