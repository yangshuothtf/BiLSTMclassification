from models import PCNN
from .BasicModule import BasicModule
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )

    def forward(self, encoder_outputs):
        # (B, L, H) -> (B , L, 1)
        energy = self.projection(encoder_outputs)
        weights = F.softmax(energy.squeeze(-1), dim=1)
        # (B, L, H) * (B, L, 1) -> (B, H)
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return outputs, weights



class LSTM(BasicModule):

    def __init__(self, opt):
        super(LSTM, self).__init__()

        print("!!!!!!!this is LSTM")

        self.opt = opt

        self.model_name = 'LSTM'

        self.word_embs = nn.Embedding(self.opt.vocab_size, self.opt.word_dim)
        self.pos1_embs = nn.Embedding(2*self.opt.pos_size + 1, self.opt.pos_dim)
        self.pos2_embs = nn.Embedding(2*self.opt.pos_size + 1, self.opt.pos_dim)
        self.relation_embeds = nn.Embedding(self.opt.rel_num, self.opt.hidden_dim)


        self.lstm = nn.LSTM(input_size=self.opt.word_dim + self.opt.pos_dim*2,
                            hidden_size=self.opt.hidden_dim//2,num_layers=1, bidirectional=True)
        # self.lstm = nn.LSTM(input_size=self.opt.word_dim,
        #                     hidden_size=self.opt.hidden_dim // 2, num_layers=1, bidirectional=True)

        # self.hidden2tag = nn.Linear(self.opt.hidden_dim, self.opt.rel_num)
        self.hidden2tag = nn.Linear(self.opt.hidden_dim//2, self.opt.rel_num)


        self.dropout_emb = nn.Dropout(p=0.5)
        self.dropout_lstm = nn.Dropout(p=0.5)
        self.dropout_att = nn.Dropout(p=0.5)

        self.hidden = self.init_hidden_lstm()
        self.attention = SelfAttention(self.opt.hidden_dim//2)

        self.fc_out = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.opt.hidden_dim//2, self.opt.hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(self.opt.hidden_dim//2, self.opt.rel_num)
        )

        self.attention_layer = nn.Sequential(
            nn.Linear(self.opt.hidden_dim//2, self.opt.hidden_dim//2),
            nn.ReLU(inplace=True)
        )
        # self.init_word_emb()
        # self.init_model_weight()


    def init_hidden_lstm(self):
        return (torch.randn(2, self.opt.batch_size, self.opt.hidden_dim // 2),
                    torch.randn(2, self.opt.batch_size, self.opt.hidden_dim // 2))






    def init_model_weight(self):
        '''
        use xavier to init
        '''
        # nn.init.xavier_normal_(self.cnn_linear.weight)
        # nn.init.constant_(self.cnn_linear.bias, 0.)
        nn.init.xavier_normal_(self.hidden2tag.weight)
        nn.init.constant_(self.hidden2tag.bias, 0.)
        # for conv in self.convs:
        #     nn.init.xavier_normal_(conv.weight)
        #     nn.init.constant_(conv.bias, 0)
        #
        # nn.init.xavier_normal(self.bilstm.all_weights[0][0], gain=np.sqrt(args.init_weight_value))
        # nn.init.xavier_normal(self.bilstm.all_weights[0][1], gain=np.sqrt(args.init_weight_value))
        # nn.init.xavier_normal(self.bilstm.all_weights[1][0], gain=np.sqrt(args.init_weight_value))
        # nn.init.xavier_normal(self.bilstm.all_weights[1][1], gain=np.sqrt(args.init_weight_value))

    def init_word_emb(self):

        w2v = torch.from_numpy(np.load(self.opt.w2v_path))

        # w2v = torch.div(w2v, w2v.norm(2, 1).unsqueeze(1))
        # w2v[w2v != w2v] = 0.0

        if self.opt.use_gpu:
            self.word_embs.weight.data.copy_(w2v.cuda())
        else:
            self.word_embs.weight.data.copy_(w2v)
    #
    # def attention_net_with_w(self, lstm_out, lstm_hidden):
    #     '''
    #
    #     :param lstm_out:    [batch_size, len_seq, n_hidden * 2]
    #     :param lstm_hidden: [batch_size, num_layers * num_directions, n_hidden]
    #     :return: [batch_size, n_hidden]
    #     '''
    #
    #     lstm_tmp_out = torch.chunk(lstm_out, 2, -1)
    #     # h [batch_size, time_step, hidden_dims]
    #     h = lstm_tmp_out[0] + lstm_tmp_out[1]
    #     # [batch_size, num_layers * num_directions, n_hidden]
    #     lstm_hidden = torch.sum(lstm_hidden, dim=1)
    #     # [batch_size, 1, n_hidden]
    #     lstm_hidden = lstm_hidden.unsqueeze(1)
    #     # atten_w [batch_size, 1, hidden_dims]
    #     atten_w = self.attention_layer(lstm_hidden)
    #     # m [batch_size, time_step, hidden_dims]
    #     m = nn.Tanh()(h)
    #     # atten_context [batch_size, 1, time_step]
    #     atten_context = torch.bmm(atten_w, m.transpose(1, 2))
    #     # softmax_w [batch_size, 1, time_step]
    #     softmax_w = F.softmax(atten_context, dim=-1)
    #     # context [batch_size, 1, hidden_dims]
    #     context = torch.bmm(softmax_w, h)
    #     result = context.squeeze(1)
    #     return result,atten_w


    def forward(self, x):

        batch_size = self.opt.batch_size

        word_feature,left_pf, right_pf= torch.chunk(x,3, dim=1)

        lengths = []
        for f in word_feature:
            lengths.append(int(max((f != 30713).nonzero()))+1)

        max_len = np.max(lengths)
        for l in lengths:
            l = l + max_len*2

        self.hidden = self.init_hidden_lstm()



        # sentence level feature
        word_emb = self.word_embs(word_feature)  # (batch_size, max_len, word_dim)
        left_emb = self.pos1_embs(left_pf)  # (batch_size, max_len, word_dim)
        right_emb = self.pos2_embs(right_pf)  # (batch_size, max_len, word_dim)


        # print(right_emb.size())
        # print(word_emb.size())
        x = torch.cat([left_emb, right_emb,word_emb], 2)  # 128 98 110
        # x = word_emb

        x = torch.transpose(x, 0, 1)  # L B E
        x = rnn_utils.pack_padded_sequence(x, lengths)
        lstm_out, (final_hidden_state, final_cell_state) = self.lstm(x, self.hidden)
        x = nn.utils.rnn.pad_packed_sequence(lstm_out)[0]

        # #hidden attention
        # x = x.permute(1, 0, 2)  # B L E
        # final_hidden_state = final_hidden_state.permute(1, 0, 2)
        # lstm_out,att_weight = self.attention_net_with_w(x,final_hidden_state)
        # return self.fc_out(lstm_out)

        #self-attention part
        x = x[:, :, :self.opt.hidden_dim//2] + x[:, :, self.opt.hidden_dim//2:]
        embedding, attn_weights = self.attention(x.transpose(0, 1))
        outputs = self.hidden2tag(embedding.view(batch_size, -1))
        return outputs,attn_weights

        #no attention part
        x = self.dropout_lstm(lstm_out)
        x = self.hidden2tag(x[-1])


        # x = torch.transpose(x, 0, 1)
        # x = torch.transpose(x, 1, 2)
        # x = torch.tanh(self.attention(x))
        # x = x.squeeze(2)
        # x = self.hidden2tag(x)

        return x