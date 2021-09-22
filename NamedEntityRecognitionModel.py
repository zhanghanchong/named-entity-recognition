import math
import torch
from torch import nn


class TokenEmbedding(nn.Module):
    def __init__(self, d_model, vocabulary_size):
        super().__init__()
        self.d_model = d_model
        self.__embedding = nn.Embedding(vocabulary_size, d_model)

    def forward(self, tokens):
        return self.__embedding(tokens) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_sequence_length):
        super().__init__()
        self.__dropout = nn.Dropout(dropout)
        positional_embedding = torch.arange(max_sequence_length).reshape(max_sequence_length, 1) / torch.pow(
            10000, torch.arange(d_model) // 2 * 2 / d_model)
        positional_embedding[:, 0::2] = torch.sin(positional_embedding[:, 0::2])
        positional_embedding[:, 1::2] = torch.cos(positional_embedding[:, 1::2])
        positional_embedding = positional_embedding.unsqueeze(1)
        self.register_buffer('positional_embedding', positional_embedding)

    def forward(self, tokens_embedding):
        return self.__dropout(tokens_embedding + self.positional_embedding[:tokens_embedding.shape[0], :])


class PretrainModel(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout, max_sequence_length, nhead, num_layers, vocabulary_size):
        super().__init__()
        self.token_embedding = TokenEmbedding(d_model, vocabulary_size)
        self.__positional_encoding = PositionalEncoding(d_model, dropout, max_sequence_length)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.__transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.__linear = nn.Linear(d_model, vocabulary_size)

    def forward(self, source, source_mask, source_padding_mask):
        tokens_embedding = self.__positional_encoding(self.token_embedding(source))
        outs = self.__transformer_encoder(tokens_embedding, source_mask, source_padding_mask)
        return self.__linear(outs)


class NamedEntityRecognitionModel(nn.Module):
    def __get_logits(self, source):
        with torch.no_grad():
            tokens_embedding = self.__token_embedding(source)
        hidden_states, _ = self.__lstm(tokens_embedding)
        return self.__linear(hidden_states)

    def __init__(self, dropout, hidden_size, num_layers, token_embedding, vocabulary_size):
        super().__init__()
        self.__vocabulary_size = vocabulary_size
        self.__token_embedding = token_embedding
        self.__lstm = nn.LSTM(input_size=token_embedding.d_model, hidden_size=hidden_size, num_layers=num_layers,
                              dropout=dropout, bidirectional=True)
        self.__linear = nn.Linear(hidden_size * 2, vocabulary_size)
        self.__transition = nn.Parameter(torch.rand((vocabulary_size, vocabulary_size)))

    def forward(self, source):
        sequence_length = source.shape[0]
        logits = self.__get_logits(source)[:, 0, :]
        opt = [[0] * self.__vocabulary_size for _ in range(sequence_length)]
        path = [[0] * self.__vocabulary_size for _ in range(sequence_length - 1)]
        for i in range(self.__vocabulary_size):
            opt[0][i] = logits[0, i].item()
        for i in range(1, sequence_length):
            for j in range(self.__vocabulary_size):
                opt[i][j] = opt[i - 1][0] + logits[i, j].item() + self.__transition[0, j].item()
                for k in range(1, self.__vocabulary_size):
                    score = opt[i - 1][k] + logits[i, j].item() + self.__transition[k, j].item()
                    if score > opt[i][j]:
                        opt[i][j] = score
                        path[i - 1][j] = k
        token_index = 0
        for i in range(1, self.__vocabulary_size):
            if opt[sequence_length - 1][i] > opt[sequence_length - 1][token_index]:
                token_index = i
        target = [token_index] * sequence_length
        for i in reversed(range(sequence_length - 1)):
            token_index = path[i][token_index]
            target[i] = token_index
        return target

    def loss_function(self, source, target, device):
        sequence_length, batch_size = source.shape
        logits = self.__get_logits(source)
        total_score = logits[0]
        for i in range(1, sequence_length):
            total_score_matrix = torch.zeros((batch_size, self.__vocabulary_size, self.__vocabulary_size),
                                             device=device)
            for j in range(self.__vocabulary_size):
                for k in range(self.__vocabulary_size):
                    total_score_matrix[:, j, k] = total_score[:, j] + logits[i, :, k] + self.__transition[j, k]
            total_score = torch.logsumexp(total_score_matrix, dim=1)
        total_score = torch.logsumexp(total_score, dim=1)
        real_path_score = torch.zeros(batch_size, device=device)
        for i in range(sequence_length):
            for j in range(batch_size):
                real_path_score[j] += logits[i, j, target[i, j]]
                if i < sequence_length - 1:
                    real_path_score[j] += self.__transition[target[i, j], target[i + 1, j]]
        return torch.mean(total_score - real_path_score)
