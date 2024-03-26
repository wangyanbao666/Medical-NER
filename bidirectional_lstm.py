import torch
from torch import nn
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class BiLSTM_CRF(nn.Module):
    def __init__(self, hidden_dim, output_size, num_layers=1, batch_first=True, dropout=0.1):
        super(BiLSTM_CRF, self).__init__()
        bert_model = 'bert-base-uncased'
        self.embedding_dim = bert_model.config.hidden_size
        self.hidden_dim = hidden_dim
        # self.tag_to_ix = tag_to_ix
        # self.target_size = len(tag_to_ix)

        # self.bert_embeds = BertModel.from_pretrained(bert_model)
        # self.lstm = nn.LSTM(embedding_dim, self.hidden_dim//2, num_layers=1, batch_first=batch_first, bidirectional=True)
        # self.dropout = nn.Dropout(dropout)
        # self.hidden2tag = nn.Linear(hidden_dim, self.target_size)
        # self.crf = CRF(self.target_size, batch_first=True)
    def get_embedding_dim(self):
        return self.embedding_dim

    def forward(self, input_ids, labels=None):
        with torch.no_grad():
            embeds = self.bert_embeds(input_ids=input_ids).last_hidden_state
      
        print("embeds size: ", embeds.shape())
        sequence_output = self.dropout(embeds)
        # h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size) # 2 for bidirectional
        # c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)
        
        # Forward propagate LSTM
        # lstm_out, (hn, cn) = self.lstm(sequence_output, None)
        # lstm_feats = self.hidden2tag(lstm_out)
        
        # if labels is not None:
        #     loss = -self.crf(lstm_feats, labels, reduction='mean')
        #     return loss
        # else:
        #     predictions = self.crf.decode(lstm_feats)
        #     return predictions
