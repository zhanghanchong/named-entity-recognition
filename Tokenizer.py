import io
import json
import torch
import os

UNK, PAD, SOS, EOS, MSK = 0, 1, 2, 3, 4
SPECIAL_TOKENS_NUM = 5


def get_dataset_filename(filename):
    return f'dataset/{filename}.txt'


def get_vocabulary_filename(filename):
    return f'vocabulary/{filename}.json'


def get_words(sentence):
    return sentence.rstrip('\n').split(' ')


class Tokenizer:
    def __build_vocabulary(self):
        word_count = {}
        with io.open(get_dataset_filename(self.__filename), 'r', encoding='UTF-8') as file:
            while 1:
                sentence = file.readline()
                if len(sentence) == 0:
                    break
                words = get_words(sentence)
                for word in words:
                    if word in word_count:
                        word_count[word] += 1
                    else:
                        word_count[word] = 1
        word_count_sorted = sorted(word_count.items(), key=lambda item: item[1], reverse=True)
        vocabulary = {'<UNK>': UNK, '<PAD>': PAD, '<SOS>': SOS, '<EOS>': EOS, '<MSK>': MSK}
        for i in range(len(word_count_sorted)):
            vocabulary[word_count_sorted[i][0]] = i + SPECIAL_TOKENS_NUM
        if not os.path.exists('vocabulary'):
            os.mkdir('vocabulary')
        with io.open(get_vocabulary_filename(self.__filename), 'w', encoding='UTF-8') as file:
            file.write(json.dumps(vocabulary, indent=4, ensure_ascii=False))

    def __init__(self, filename):
        self.__filename = filename
        self.__file = None
        if not os.path.exists(get_vocabulary_filename(filename)):
            self.__build_vocabulary()
        with io.open(get_vocabulary_filename(filename), 'r', encoding='UTF-8') as file:
            self.word_index = json.load(file)
        self.index_word = []
        for word in self.word_index:
            self.index_word.append(word)

    def get_sequence(self, words, sequence_length):
        sequence = torch.zeros((sequence_length, 1), dtype=torch.int64)
        sequence[0, 0] = SOS
        for i in range(len(words)):
            if words[i] in self.word_index:
                sequence[i + 1, 0] = self.word_index[words[i]]
            else:
                sequence[i + 1, 0] = UNK
        sequence[len(words) + 1, 0] = EOS
        for i in range(len(words) + 2, sequence_length):
            sequence[i, 0] = PAD
        return sequence

    def get_batch(self, batch_size):
        if self.__file is None:
            self.__file = io.open(get_dataset_filename(self.__filename), 'r', encoding='UTF-8')
        words_list = []
        sequence_length = 0
        for _ in range(batch_size):
            sentence = self.__file.readline()
            if len(sentence) == 0:
                break
            words = get_words(sentence)
            words_list.append(words)
            sequence_length = max(sequence_length, len(words))
        batch_size = len(words_list)
        if batch_size == 0:
            self.__file.close()
            self.__file = None
            return None
        sequence_length += 2
        batch = torch.zeros((sequence_length, batch_size), dtype=torch.int64)
        for i in range(batch_size):
            batch[:, i] = self.get_sequence(words_list[i], sequence_length)[:, 0]
        return batch
