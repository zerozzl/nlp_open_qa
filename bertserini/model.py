import torch
from torch import nn
from torch.nn.utils import rnn
from transformers import BertConfig, BertModel


class Bert(nn.Module):
    def __init__(self, config_path, model_path, bert_freeze):
        super(Bert, self).__init__()

        config = BertConfig.from_json_file(config_path)
        self.embedding = BertModel.from_pretrained(model_path, config=config)
        self.linear = nn.Linear(config.hidden_size, 2)

        if bert_freeze:
            for param in self.embedding.parameters():
                param.requires_grad = False

        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, tokens, segments, decode=True, tags=None):
        tokens = rnn.pad_sequence(tokens, batch_first=True)
        segments = rnn.pad_sequence(segments, batch_first=True)
        masks = (tokens > 0).int()

        out = self.embedding(input_ids=tokens, token_type_ids=segments, attention_mask=masks)
        out = out.last_hidden_state
        out = self.linear(out)

        start = out[:, :, 0]
        end = out[:, :, 1]

        if decode:
            start_idx = torch.argmax(start, dim=1)
            end_idx = torch.argmax(end, dim=1)
            return start, end, start_idx, end_idx
        else:
            tags_start = tags[:, 0]
            tags_end = tags[:, 1]

            loss_start = self.ce_loss(start, tags_start)
            loss_end = self.ce_loss(end, tags_end)

            loss = loss_start + loss_end
            return loss
