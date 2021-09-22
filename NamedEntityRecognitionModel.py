import math
import torch
from torch import nn


class TokenEmbedding(nn.Module):
    def __init__(self, d_model, vocabulary_size):
        super().__init__()
        self.__d_model = d_model
        self.__embedding = nn.Embedding(vocabulary_size, d_model)

    def forward(self, tokens):
        return self.__embedding(tokens) * math.sqrt(self.__d_model)


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


class NamedEntityRecognitionModel(nn.Module):
    def __init__(self, d_model, dropout, hidden_size, num_layers, vocabulary_size_source, vocabulary_size_target):
        super().__init__()
        self.__vocabulary_size_target = vocabulary_size_target
        self.__token_embedding = TokenEmbedding(d_model, vocabulary_size_source)
        self.__lstm = nn.LSTM(input_size=d_model, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,
                              bidirectional=True)
        self.__linear = nn.Linear(hidden_size * 2, vocabulary_size_target)
        self.__transition = nn.Parameter(torch.rand((vocabulary_size_target, vocabulary_size_target)))

    def loss_function(self, source, target, device):
        sequence_length, batch_size = source.shape
        hidden_states, _ = self.__lstm(self.__token_embedding(source))
        logits = self.__linear(hidden_states)
        total_score = logits[0]
        for i in range(1, sequence_length):
            total_score_matrix = torch.zeros((batch_size, self.__vocabulary_size_target, self.__vocabulary_size_target),
                                             device=device)
            for j in range(self.__vocabulary_size_target):
                for k in range(self.__vocabulary_size_target):
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
