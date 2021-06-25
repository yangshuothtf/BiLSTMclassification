# -*- coding: utf-8 -*-

from config import opt
import models
import dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
import time
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import numpy as np
import random
import torch.utils.data as data
import torch.nn.utils.rnn as rnn_utils
import operator
from IPython.display import HTML, display
import matplotlib.pyplot as plt



def now():
    return str(time.strftime('%Y-%m-%d %H:%M%S'))


def test(**kwargs):
    pass


def highlight(word, attn):
    html_color = '#%02X%02X%02X' % (255, int(255*(1 - attn)), int(255*(1 - attn)))
    return '<span style="background-color: {}">{}</span>'.format(html_color, word)

def mk_html(seq, attns):
    html = ""
    for ix, attn in zip(seq, attns):
        html += ' ' + highlight(
            ix,
            attn
        )
    return html + "<br><br>\n"


def collate_fn(input):

    # data,label = input
    data = [i[0] for i in input]
    left = [i[1] for i in input]
    right = [i[2] for i in input]
    label = [i[3] for i in input]

    # data.sort(key=lambda x: len(x), reverse=True)

    data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=9712)
    left = rnn_utils.pad_sequence(left, batch_first=True, padding_value=0)
    right = rnn_utils.pad_sequence(right, batch_first=True, padding_value=0)
    return data,left,right,label


def train(**kwargs):

    Test_Lost = []
    Train_Lost = []
    Accuracy = []
    F1 = []
    best_epoch = 0

    print("Start Train")
    opt.parse(kwargs)
    if opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)

    # loading data
    DataModel = getattr(dataset, 'SEMData')
    train_data = DataModel(opt.data_root, train=True)
    print(train_data)
    train_data_loader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)

    test_data = DataModel(opt.data_root, train=False)
    test_data_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    print('train data: {}; test data: {}'.format(len(train_data), len(test_data)))

    # criterion and optimizer
    # lr = opt.lr
    model = getattr(models, 'PCNN')(opt)
    if opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    #  optimizer = optim.Adam(model.out_linear.parameters(), lr=0.0001)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # optimizer = optim.Adadelta(model.parameters(), rho=0.95, eps=1e-6)

    best_acc = 0.0
    # train
    for epoch in range(opt.num_epochs):

        total_loss = 0.0

        for ii, data in enumerate(train_data_loader):
            if opt.use_gpu:
                data = list(map(lambda x: Variable(x.cuda()), data))
            else:
                data = list(map(Variable, data))

            model.zero_grad()
            out = model(data[:-1])
            loss = criterion(out, data[-1])
            loss.backward()
            optimizer.step()
            total_loss += loss.data.item()

        train_avg_loss = total_loss / len(train_data_loader.dataset)
        acc, f1, eval_avg_loss, pred_y = eval(model, test_data_loader, opt.rel_num)
        if best_acc < acc:
            best_acc = acc
            best_epoch = ii
            write_result(model.model_name, pred_y)
            model.save(name="SEM_CNN")
        # toy_acc, toy_f1, toy_loss = eval(model, train_data_loader, opt.rel_num)
        print('Epoch {}/{}: train loss: {}; test accuracy: {}, test f1:{},  test loss {}'.format(
            epoch, opt.num_epochs, train_avg_loss, acc, f1, eval_avg_loss))
        Accuracy.append(acc)
        F1.append(f1)
        Train_Lost.append(train_avg_loss)
        Test_Lost.append(eval_avg_loss)

    print("*" * 30)
    print("the best epoch:", best_epoch)
    print("the best acc: {};".format(best_acc))
    print("*" * 30)
    x_epoch = range(opt.num_epochs)

    plt.plot(x_epoch, Accuracy, label='Accuracy')
    plt.plot(x_epoch, F1, label='F1')
    plt.plot(x_epoch, Train_Lost, label='Train Loss')
    plt.plot(x_epoch, Test_Lost, label='Test Loss')
    plt.xlabel("epoch")

    plt.legend()
    plt.show()



def lstm_train(**kwargs):

    Test_Lost = []
    Train_Lost = []
    Accuracy = []
    Html_List= []
    F1  =[]
    best_epoch=0

    print("LSTM Start Train")
    opt.parse(kwargs)
    if opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)

    # loading data
    DataModel = getattr(dataset, 'SEMData')
    train_data = DataModel(opt.data_root, train=True)
    train_data_loader = DataLoader(train_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers,drop_last=True,collate_fn=collate_fn)

    test_data = DataModel(opt.data_root, train=False)
    test_data_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers,drop_last=True,collate_fn=collate_fn)
    print('train data: {}; test data: {}'.format(len(train_data), len(test_data)))


    # criterion and optimizer
    # lr = opt.lr
    model = getattr(models, 'LSTM')(opt)
    if opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    #  optimizer = optim.Adam(model.out_linear.parameters(), lr=0.0001)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # optimizer = optim.Adadelta(model.parameters(), rho=0.95, eps=1e-6)

    best_acc = 0.0
    # train
    for epoch in range(opt.num_epochs):

        total_loss = 0.0

        for ii, data in enumerate(train_data_loader):
            # if opt.use_gpu:
            #     data = list(map(lambda x: Variable(x.cuda()), data))
            # else:
            #     data = list(map(Variable, data))
            x = data[0]
            left = data[1]
            right = data[2]
            y=torch.LongTensor(data[3])

            model.zero_grad()
            # out= model(torch.cat((x,left,right),dim=1))
            out,att_weight = model(torch.cat((x,left,right),dim=1))
            # out = model(x)
            loss = criterion(out, y)

            loss.backward()
            optimizer.step()
            total_loss += loss.data.item()

        train_avg_loss = total_loss / len(train_data_loader.dataset)
        html , acc, f1, eval_avg_loss, pred_y = lstm_eval(model, test_data_loader, opt.rel_num)
        if best_acc < acc:
            best_acc = acc
            best_epoch = epoch
            write_result(model.model_name, pred_y)
            model.save(name="SEM_LSTM")
        # toy_acc, toy_f1, toy_loss = eval(model, train_data_loader, opt.rel_num)
        print('Epoch {}/{}: train loss: {}; test accuracy: {}, test f1:{},  test loss {}'.format(
            epoch, opt.num_epochs, train_avg_loss, acc, f1, eval_avg_loss))
        Accuracy.append(acc)
        F1.append(f1)
        Train_Lost.append(train_avg_loss)
        Test_Lost.append(eval_avg_loss)
        Html_List.append(html)


    print("*" * 30)
    print("the best epoch:",best_epoch)
    print("the best acc: {};".format(best_acc))
    print("*" * 30)
    x_epoch = range(opt.num_epochs)


    plt.plot(x_epoch, Accuracy, label='Accuracy')
    plt.plot(x_epoch, F1, label='F1')
    plt.plot(x_epoch, Train_Lost, label='Train Loss')
    plt.plot(x_epoch, Test_Lost, label='Test Loss')
    plt.xlabel("epoch")

    plt.legend()
    plt.show()

    display(HTML('<h1>一个标题</h1>') )

    for h in Html_List:
        for l in h:
            display(l)
        print("*" * 30)






id2word = np.load('dataset/SemEval/train/npy/id2word.npy')
id2word = id2word.item()

def lstm_eval(model, test_data_loader, k):


    model.eval()
    avg_loss = 0.0
    pred_y = []
    labels = []
    HTML_List=[]

    for ii, data in enumerate(test_data_loader):

        # if opt.use_gpu:
        #     data = list(map(lambda x: torch.LongTensor(x).cuda(), data))
        # else:
        #     data = list(map(lambda x: torch.LongTensor(x), data))

        x = data[0]
        left = data[1]
        right = data[2]
        y = torch.LongTensor(data[3])

        out,att_weight = model(torch.cat((x,left,right),dim=1))
        # out = model(x)
        # out = model(torch.cat((x, left, right), dim=1))  
        loss = F.cross_entropy(out, y)

        pred_y.extend(torch.max(out, 1)[1].data.cpu().numpy().tolist())
        labels.extend(y.data.cpu().numpy().tolist())
        avg_loss += loss.data.item()

        # display(HTML("<h1>这是一个标题</h1>"))
        if ii<5 or ii == opt.batch_size-1:
            for idx in range(0,opt.batch_size):
                text = list(map(lambda x: id2word.get(int(x)), x[idx]))
                html_text = mk_html(text, att_weight[idx])
                HTML_List.append(HTML(html_text))

    # size = len(test_data_loader.dataset) - len(test_data_loader.dataset) % opt.batch_size
    assert len(pred_y) ==  len(labels)
    f1 = f1_score(labels, pred_y, average='macro')
    acc = accuracy_score(labels, pred_y)
    model.train()
    return HTML_List,acc, f1, avg_loss /len(pred_y), pred_y



def eval(model, test_data_loader, k):

    model.eval()
    avg_loss = 0.0
    pred_y = []
    labels = []
    for ii, data in enumerate(test_data_loader):

        if opt.use_gpu:
            data = list(map(lambda x: torch.LongTensor(x).cuda(), data))
        else:
            data = list(map(lambda x: torch.LongTensor(x), data))

        out = model(data[:-1])
        loss = F.cross_entropy(out, data[-1])

        pred_y.extend(torch.max(out, 1)[1].data.cpu().numpy().tolist())
        labels.extend(data[-1].data.cpu().numpy().tolist())
        avg_loss += loss.data.item()

    # size = len(test_data_loader.dataset) - len(test_data_loader.dataset) % opt.batch_size
    assert len(pred_y) ==  len(labels)
    f1 = f1_score(labels, pred_y, average='macro')
    acc = accuracy_score(labels, pred_y)
    model.train()
    return acc, f1, avg_loss /len(pred_y), pred_y





def write_result(model_name, pred_y):
    out = open('./semeval/sem_{}_result.txt'.format(model_name), 'w')
    size = len(pred_y)
    start = 8001
    end = start + size
    for i in range(start, end):
        out.write("{}\t{}\n".format(i, pred_y[i - start]))


if __name__ == "__main__":
    #     import fire
    # fire.Fire()
    # train()

    lstm_train()