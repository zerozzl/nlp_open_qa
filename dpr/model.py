import torch
from torch import nn
from torch.nn.utils import rnn
import torch.nn.functional as F
from transformers import BertConfig, BertModel


class Retriever(nn.Module):
    def __init__(self, config_path, model_path, bert_freeze):
        super(Retriever, self).__init__()

        config = BertConfig.from_json_file(config_path)
        self.question_encoder = BertModel.from_pretrained(model_path, config=config)
        self.context_encoder = BertModel.from_pretrained(model_path, config=config)

        if bert_freeze:
            for param in self.question_encoder.parameters():
                param.requires_grad = False
            for param in self.context_encoder.parameters():
                param.requires_grad = False

        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, questions, questions_seg, contexts, contexts_seg, decode=True, tags=None):
        questions = rnn.pad_sequence(questions, batch_first=True)
        questions_seg = rnn.pad_sequence(questions_seg, batch_first=True)
        questions_mask = (questions > 0).int()
        contexts = rnn.pad_sequence(contexts, batch_first=True)
        contexts_seg = rnn.pad_sequence(contexts_seg, batch_first=True)
        contexts_mask = (contexts > 0).int()

        questions_rep = self.question_encoder(input_ids=questions,
                                              token_type_ids=questions_seg,
                                              attention_mask=questions_mask)
        questions_rep = questions_rep.last_hidden_state
        questions_rep = questions_rep[:, 0, :]

        contexts_rep = self.context_encoder(input_ids=contexts,
                                            token_type_ids=contexts_seg,
                                            attention_mask=contexts_mask)
        contexts_rep = contexts_rep.last_hidden_state
        contexts_rep = contexts_rep[:, 0, :]

        scores = torch.matmul(questions_rep, torch.transpose(contexts_rep, 0, 1))

        if decode:
            scores = F.softmax(scores, dim=1)
            preds = torch.argmax(scores, dim=1)
            return scores, preds
        else:
            loss = self.ce_loss(scores, tags)
            return loss

    def get_question_embedding(self, questions, questions_seg):
        questions_mask = (questions > 0).int()
        questions_rep = self.question_encoder(input_ids=questions,
                                              token_type_ids=questions_seg,
                                              attention_mask=questions_mask)
        questions_rep = questions_rep.last_hidden_state
        questions_rep = questions_rep[:, 0, :]
        return questions_rep

    def get_context_embedding(self, contexts, contexts_seg):
        contexts_mask = (contexts > 0).int()
        contexts_rep = self.context_encoder(input_ids=contexts,
                                            token_type_ids=contexts_seg,
                                            attention_mask=contexts_mask)
        contexts_rep = contexts_rep.last_hidden_state
        contexts_rep = contexts_rep[:, 0, :]
        return contexts_rep
