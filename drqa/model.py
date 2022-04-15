import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import rnn


class AlignedEmbedding(nn.Module):
    def __init__(self, embed_size):
        super(AlignedEmbedding, self).__init__()
        self.linear = nn.Linear(embed_size, embed_size)

    def forward(self, x, y, y_mask):
        x_embed = F.relu(self.linear(x))
        y_embed = F.relu(self.linear(y))

        scores = x_embed.bmm(y_embed.transpose(2, 1))
        y_mask = y_mask.unsqueeze(1).expand(scores.size())
        scores.data.masked_fill_(y_mask.data, -float('inf'))

        alpha = F.softmax(scores, dim=-1)
        alpha_embed = alpha.bmm(y)

        return alpha_embed


class SelfAttentionLayer(nn.Module):
    def __init__(self, input_size):
        super(SelfAttentionLayer, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x, x_mask):
        scores = self.linear(x).squeeze(-1)
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores, dim=1)
        return alpha


class CrossAttentionLayer(nn.Module):
    def __init__(self, x_size, y_size):
        super(CrossAttentionLayer, self).__init__()
        self.linear = nn.Linear(y_size, x_size)

    def forward(self, x, y, x_mask):
        y_embed = self.linear(y)
        scores = x.bmm(y_embed.unsqueeze(2)).squeeze(-1)
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        return scores


class DocumentReader(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size, hidden_layer_num, inp_dropout_rate, hid_dropout_rate,
                 embed_fix=False, use_bigram=False, bigram_vocab_size=0, bigram_embed_size=0):
        super(DocumentReader, self).__init__()
        self.use_bigram = use_bigram

        inp_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)

        if use_bigram:
            inp_size += bigram_embed_size * 2
            self.bigram_embedding = nn.Embedding(bigram_vocab_size, bigram_embed_size)

        if embed_fix:
            for param in self.embedding.parameters():
                param.requires_grad = False

            if use_bigram:
                for param in self.bigram_embedding.parameters():
                    param.requires_grad = False

        self.in_dropout = nn.Dropout(inp_dropout_rate)

        self.aligned_embedding = AlignedEmbedding(embed_size)

        self.question_rnn = nn.LSTM(input_size=inp_size,
                                    hidden_size=hidden_size,
                                    num_layers=hidden_layer_num,
                                    dropout=hid_dropout_rate,
                                    batch_first=True,
                                    bidirectional=True)

        self.context_rnn = nn.LSTM(input_size=inp_size + embed_size + 1,
                                   hidden_size=hidden_size,
                                   num_layers=hidden_layer_num,
                                   dropout=hid_dropout_rate,
                                   batch_first=True,
                                   bidirectional=True)

        self.question_attention = SelfAttentionLayer(hidden_size * 2)

        self.start_attn = CrossAttentionLayer(hidden_size * 2, hidden_size * 2)
        self.end_attn = CrossAttentionLayer(hidden_size * 2, hidden_size * 2)

        self.ce_loss = nn.CrossEntropyLoss()

    def init_embedding(self, pretrained_embeddings):
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

    def init_bigram_embedding(self, pretrained_embeddings):
        self.bigram_embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

    def forward(self, question, context, ans_ctx_ext_match, question_bigram=None, ans_ctx_bigram=None,
                decode=True, tags=None):
        context_len = [len(item) for item in context]
        question = rnn.pad_sequence(question, batch_first=True)
        context = rnn.pad_sequence(context, batch_first=True)
        ans_ctx_ext_match = rnn.pad_sequence(ans_ctx_ext_match, batch_first=True)
        question_bigram = rnn.pad_sequence(question_bigram, batch_first=True)
        ans_ctx_bigram = rnn.pad_sequence(ans_ctx_bigram, batch_first=True)

        context_mask = context == 0
        question_mask = question == 0
        ans_ctx_ext_match = ans_ctx_ext_match.unsqueeze(-1)

        q_embed = self.embedding(question)
        c_embed = self.embedding(context)
        aligned_embed = self.aligned_embedding(c_embed, q_embed, question_mask)

        if self.use_bigram:
            q_embed_bi = torch.cat(
                [self.bigram_embedding(question_bigram[:, :, i]) for i in range(question_bigram.size()[2])], dim=2)
            q_embed = torch.cat((q_embed, q_embed_bi), dim=2)

            c_embed_bi = torch.cat(
                [self.bigram_embedding(ans_ctx_bigram[:, :, i]) for i in range(ans_ctx_bigram.size()[2])], dim=2)
            c_embed = torch.cat((c_embed, c_embed_bi), dim=2)

        q_embed = self.in_dropout(q_embed)
        c_embed = self.in_dropout(c_embed)
        aligned_embed = self.in_dropout(aligned_embed)

        c_embed = torch.cat([c_embed, aligned_embed, ans_ctx_ext_match], dim=2)
        c_embed = rnn.pack_padded_sequence(c_embed, context_len, batch_first=True)
        q_rep, _ = self.question_rnn(q_embed)
        c_rep, _ = self.context_rnn(c_embed)
        c_rep, _ = rnn.pad_packed_sequence(c_rep, batch_first=True)

        q_attn = self.question_attention(q_rep, question_mask)
        q_rep = q_attn.unsqueeze(1).bmm(q_rep).squeeze(1)

        start_scores = self.start_attn(c_rep, q_rep, context_mask)
        end_scores = self.end_attn(c_rep, q_rep, context_mask)

        if decode:
            start_p = torch.exp(start_scores)
            end_p = torch.exp(end_scores)
            start_idx = torch.argmax(start_p, dim=-1)
            end_idx = torch.argmax(end_p, dim=-1)
            return start_p, end_p, start_idx, end_idx
        else:
            tags_start = tags[:, 0]
            tags_end = tags[:, 1]

            loss_start = self.ce_loss(start_scores, tags_start)
            loss_end = self.ce_loss(end_scores, tags_end)

            loss = loss_start + loss_end
            return loss
