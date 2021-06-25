# -*- coding: utf-8 -*-

from torch.utils.data import Dataset
import os
import numpy as np
import string
import torch

lstm_not_padding = True

class SEMData(Dataset):

    def __init__(self, root_path, train=True):
        if train:
            path = os.path.join(root_path, 'train/npy/')
            print('loading train data')
        else:
            path = os.path.join(root_path, 'test/npy/')
            print('loading test data')


        if lstm_not_padding:
            self.word_feature = np.load(path + 'word_feautre.npy')
            # self.lexical_feature = np.load(path + 'lexical_feature.npy')
            self.right_pf = np.load(path + 'right_pf.npy')
            self.left_pf = np.load(path + 'left_pf.npy')
            self.labels = np.load(path + 'labels.npy')
        else:
            self.word_feature = np.load(path + 'word_feautre.npy')
            self.lexical_feature = np.load(path + 'lexical_feature.npy')
            self.right_pf = np.load(path + 'right_pf.npy')
            self.left_pf = np.load(path + 'left_pf.npy')
            self.labels = np.load(path + 'labels.npy')
            self.x = list(zip(self.lexical_feature, self.word_feature, self.left_pf, self.right_pf, self.labels))
            print('loading finish')


    def __getitem__(self, idx):
        # assert idx < len(self.x)
        return torch.LongTensor(self.word_feature[idx]),torch.LongTensor(self.left_pf[idx]),torch.LongTensor(self.right_pf[idx]),self.labels[idx]

    def __len__(self):
        return len(self.word_feature)


class SEMLoad(object):
    '''
    load and preprocess data
    '''
    def __init__(self, root_path, train=True, max_len=98, limit=50):

        self.stoplists = set(string.punctuation)

        self.max_len = max_len
        self.limit = limit
        self.root_path = root_path
        self.train = train
        if self.train:
            print('train data:')
        else:
            print('test data:')


        # self.rel_path = os.path.join(root_path, 'relation2id.txt')
        self.w2v_path = os.path.join(root_path, 'Corpus/corpus_vectors_100.txt')
        self.train_path = os.path.join(root_path, 'Corpus/corpus_train_modify0.txt')
        self.vocab_path = os.path.join(root_path, 'Corpus/corpus_vocab_100.txt')
        self.test_path = os.path.join(root_path, 'Corpus/corpus_test_modify0.txt')

        print('loading start....')
        # self.rel2id, self.id2rel = self.load_rel()
        self.w2v, self.word2id, self.id2word = self.load_w2v()

        if train and lstm_not_padding == False :
            self.lexical_feature, sen_feature, self.labels = self.parse_sen(self.train_path)
            self.word_feature, self.left_pf, self.right_pf = sen_feature
        elif train == False and lstm_not_padding == False:
            self.lexical_feature, sen_feature, self.labels = self.parse_sen(self.test_path)
            self.word_feature, self.left_pf, self.right_pf = sen_feature
        elif train and lstm_not_padding :
            self.lexical_feature, self.word_feature, self.left_pf, self.right_pf, self.labels = self.parse_sen(self.train_path)
            
            # for idx in range(0,len(self.lexical_feature)):
            #     self.lexical_feature[idx].extend(self.word_feature[idx])
            # self.word_feature = self.lexical_feature

            # self.word_feature = torch.cat(torch.LongTensor(self.lexical_feature),torch.LongTensor(self.word_feature)).numpy().tolist()

        elif train == False and lstm_not_padding :
            self.lexical_feature, self.word_feature, self.left_pf, self.right_pf, self.labels = self.parse_sen(self.test_path)
            # for idx in range(0,len(self.lexical_feature)):
            #     self.lexical_feature[idx].extend(self.word_feature[idx])
            # self.word_feature = self.lexical_feature

        print('loading finish')

    def save(self):
        if self.train:
            prefix = 'train'
        else:
            prefix = 'test'
        np.save(os.path.join(self.root_path, prefix, 'npy/word_feautre.npy'), self.word_feature)
        np.save(os.path.join(self.root_path, prefix, 'npy/left_pf.npy'), self.left_pf)
        np.save(os.path.join(self.root_path, prefix, 'npy/right_pf.npy'), self.right_pf)
        np.save(os.path.join(self.root_path, prefix, 'npy/lexical_feature.npy'), self.lexical_feature)
        np.save(os.path.join(self.root_path, prefix, 'npy/labels.npy'), self.labels)
        np.save(os.path.join(self.root_path, prefix, 'npy/w2v.npy'), self.w2v)
        np.save(os.path.join(self.root_path, prefix, 'npy/id2word.npy'), self.id2word)
        print('save finish!')

    def load_rel(self):
        '''
        load relations
        '''
        rels = [i.strip('\n').split() for i in open(self.rel_path)]
        rel2id = {j: int(i) for i, j in rels}
        id2rel = {int(i): j for i, j in rels}

        return rel2id, id2rel

    def load_w2v(self):
        '''
        reading from vec.bin
        add two extra tokens:
            : UNK for unkown tokens
            : BLANK for the max len sentence
        '''
        wordlist = []
        vecs = []

        w2v = open(self.w2v_path)
        for line in w2v:
            line = line.strip('\n').split()
            word = line[0]
            vec = list(map(float, line[1:]))
            wordlist.append(word)
            vecs.append(np.array(vec))

        # wordlist.append('UNK')
        # wordlist.append('BLANK')
        # vecs.append(np.random.normal(size=dim, loc=0, scale=0.05))
        # vecs.append(np.random.normal(size=dim, loc=0, scale=0.05))
        # vecs.append(np.zeros(dim))
        # vecs.append(np.zeros(dim))
        word2id = {j: i for i, j in enumerate(wordlist)}
        id2word = {i: j for i, j in enumerate(wordlist)}

        return np.array(vecs, dtype=np.float32), word2id, id2word

    def parse_sen(self, path):
        '''
        parse the records in data
        '''
        all_sens =[]
        all_labels =[]
        for line in open(path, 'r'):
            line = line.strip('\n').split(' ')
            sens = line[5:]
            print(line)
            rel = int(line[0])

            ent1 = (int(line[1]), int(line[2]))
            ent2 = (int(line[3]), int(line[4]))

            all_labels.append(rel)
            sens = list(map(lambda x: self.word2id.get(x, self.word2id['<UNK>']), sens))

            all_sens.append((ent1, ent2, sens))


        lexical_feature = self.get_lexical_feature(all_sens)
        if lstm_not_padding == False:
            sen_feature = self.get_sentence_feature(all_sens)
            return lexical_feature, sen_feature, all_labels  
        else:
            word_feature,left,right = self.get_sentence_feature(all_sens)

            sorted_lexical =[[]]*len(word_feature)
            sorted_x = [0]*len(word_feature)
            sorted_y = [0]*len(word_feature)

            sorted_left = [0]*len(word_feature)
            sorted_right= [0]*len(word_feature)


            sen_len = []
            for sen in word_feature:
                sen_len.append(len(sen))
            sen_index = np.argsort(sen_len)
            for idx, l in enumerate(sen_index):
                sorted_lexical[len(sorted_x)-idx-1] = lexical_feature[l]
                sorted_x[len(sorted_x)-idx-1] = word_feature[l]
                sorted_y[len(sorted_x)-idx-1] = all_labels[l]
                sorted_left[len(sorted_x) - idx - 1] = left[l]
                sorted_right[len(sorted_x) - idx - 1] = right[l]
            # print(sorted_x)
            # for idx in range(0,len(sorted_x)):
            #     test = ([len(sorted_x[idx])]*12)+sorted_left[idx]
            #     sorted_left[idx] = test
            #     test = ([len(sorted_x[idx])] * 12) + sorted_right[idx]
            #     sorted_right[idx] = test

            return sorted_lexical, sorted_x,sorted_left,sorted_right, sorted_y  

    def get_lexical_feature(self, sens):
        '''
        : noun1
        : noun2
        : left and right tokens of noun1
        : left and right tokens of noun2
        : # WordNet hypernyms
        '''

        lexical_feature = []
        for idx, sen in enumerate(sens):
            pos_e1, pos_e2, sen = sen
            left_e1 = self.get_left_word(pos_e1, sen)
            left_e2 = self.get_left_word(pos_e2, sen)
            right_e1 = self.get_right_word(pos_e1, sen)
            right_e2 = self.get_right_word(pos_e2, sen)
            e1 = sen[pos_e1[0]:pos_e1[1]+1]
            e2 = sen[pos_e2[0]:pos_e2[1]+1]

            e1.extend(e1*3)
            e2.extend(e2*3)

            if len(e1)<=4:
                e1.extend([self.word2id['<PAD>']]*(4-len(e1)))
            else:
                e1 = e1[0:4]
            if len(e2)<=4:
                e2.extend([self.word2id['<PAD>']]*(4-len(e2)))
            else:
                e2 = e2[0:4]


            lexical_feature.append([e1[0],e1[1],e1[2],e1[3],
                                    left_e1, right_e1,
                                    e2[0],e2[1], e2[2], e2[3],
                                    left_e2, right_e2])
            #lexical_feature.append([sen[pos_e1[0]], left_e1, right_e1, sen[pos_e2[0]], left_e2, right_e2])

        print(lexical_feature)
        return lexical_feature

    def get_sentence_feature(self, sens):
        '''
        : word embedding
        : postion embedding
        return:
        sen list
        pos_left
        pos_right
        '''
        update_sens = []
        left=[]
        right=[]
        final_sens = []

        for sen in sens:
            pos_e1, pos_e2, sen = sen
            pos_left = []
            pos_right = []
            ori_len = len(sen)
            for idx in range(ori_len):
                p1 = self.get_pos_feature(idx - pos_e1[0])
                p2 = self.get_pos_feature(idx - pos_e2[0])
                if lstm_not_padding:
                    p1 = p1 + ori_len
                    p2 = p2 + ori_len
                pos_left.append(p1)
                pos_right.append(p2)
            if ori_len > self.max_len:
                sen = sen[: self.max_len]
                pos_left = pos_left[: self.max_len]
                pos_right = pos_right[: self.max_len]
            elif ori_len < self.max_len and lstm_not_padding == False:
                sen.extend([self.word2id['<PAD>']] * (self.max_len - ori_len))
                pos_left.extend([self.limit * 2 + 2] * (self.max_len - ori_len))
                pos_right.extend([self.limit * 2 + 2] * (self.max_len - ori_len))

            update_sens.append([sen, pos_left, pos_right])
            left.append(pos_left)
            right.append(pos_right)
            final_sens.append(sen)

        if lstm_not_padding == False:
            return zip(*update_sens)
        else:
            return final_sens, left, right

    def get_left_word(self, pos, sen):
        '''
        get the left word id of the token of position
        '''
        pos = pos[0]
        if pos > 0:
            return sen[pos - 1]

        else:
            # return sen[pos]
            return self.word2id['<PAD>']

    def get_right_word(self, pos, sen):
        '''
        get the right word id of the token of position
        '''
        pos = pos[1]
        if pos < len(sen) - 1:
            return sen[pos + 1]
        else:
            # return sen[pos]
            return self.word2id['<PAD>']

    def get_pos_feature(self, x):
        '''
        clip the postion range:
        : -limit ~ limit => 0 ~ limit * 2+2
        : -51 => 0
        : -50 => 1
        : 50 => 101
        : >50: 102
        '''
        if lstm_not_padding:
           return x

        if x < -self.limit:
            return 0
        if -self.limit <= x <= self.limit:
            return x + self.limit + 1
        if x > self.limit:
            return self.limit * 2 + 1
        return x


if __name__ == "__main__":
    print("！！！！！！！！！！！！！！！！！！！")
    data = SEMLoad('./SemEval/', train=True)

    data.save()
    data = SEMLoad('./SemEval/', train=False)
    data.save()
