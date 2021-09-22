import io
import json
import os
import torch
import wx
from torch import optim
from NamedEntityRecognitionModel import NamedEntityRecognitionModel
from Tokenizer import Tokenizer

MAX_SEQUENCE_LENGTH = 5000
PRETRAIN_MODEL_FILENAME = 'model/pretrain.pth'
FINETUNE_MODEL_FILENAME = 'model/finetune.pth'


class Gui(wx.Frame):
    def __set_parameters(self):
        with io.open('parameters.json', 'r') as file:
            parameters = json.load(file)
        self.__adam_beta1 = parameters['adam_beta1']
        self.__adam_beta2 = parameters['adam_beta2']
        self.__adam_epsilon = parameters['adam_epsilon']
        self.__batch_size = parameters['batch_size']
        self.__d_model = parameters['d_model']
        self.__dim_feedforward = parameters['dim_feedforward']
        self.__dropout = parameters['dropout']
        self.__hidden_size = parameters['hidden_size']
        self.__learning_rate = parameters['learning_rate']
        self.__nhead = parameters['nhead']
        self.__num_encoder_layers = parameters['num_encoder_layers']
        self.__num_lstm_layers = parameters['num_lstm_layers']
        self.__device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __pretrain(self, event):
        pass

    def __train(self, event):
        epochs = self.__text_ctrl_train_epochs.GetValue()
        try:
            epochs = int(epochs)
        except:
            self.__text_ctrl_train_logs.SetValue('"Epochs" should be a positive integer.')
            return
        if epochs <= 0:
            self.__text_ctrl_train_logs.SetValue('"Epochs" should be a positive integer.')
            return
        self.__text_ctrl_train_logs.Clear()
        tokenizer = {'text': Tokenizer('text'), 'label': Tokenizer('label')}
        if os.path.exists(FINETUNE_MODEL_FILENAME):
            model = torch.load(FINETUNE_MODEL_FILENAME)
        else:
            model = NamedEntityRecognitionModel(self.__d_model, self.__dropout,
                                                self.__hidden_size, self.__num_lstm_layers,
                                                len(tokenizer['text'].index_word), len(tokenizer['label'].index_word))
        model = model.to(self.__device)
        model.train()
        optimizer = optim.Adam(model.parameters(), self.__learning_rate, (self.__adam_beta1, self.__adam_beta2),
                               self.__adam_epsilon)
        for epoch in range(epochs):
            batch = 0
            while 1:
                source = tokenizer['text'].get_batch(self.__batch_size)
                target = tokenizer['label'].get_batch(self.__batch_size)
                if source is None and target is None:
                    break
                source = source.to(self.__device)
                target = target.to(self.__device)
                optimizer.zero_grad()
                loss = model.loss_function(source, target, self.__device)
                loss.backward()
                optimizer.step()
                batch += 1
                self.__text_ctrl_train_logs.AppendText(f'epoch {epoch + 1} batch {batch} loss {loss.item():.4f}\n')
        if not os.path.exists('model'):
            os.makedirs('model')
        torch.save(model, FINETUNE_MODEL_FILENAME)

    def __predict(self, event):
        pass

    def __init__(self):
        super().__init__(None, title='Translator', size=(600, 600))
        self.Center()
        panel = wx.Panel(self)
        self.__text_ctrl_train_epochs = wx.TextCtrl(panel)
        self.__text_ctrl_train_logs = wx.TextCtrl(panel, style=wx.TE_MULTILINE)
        self.__text_ctrl_predict_source_sentences = wx.TextCtrl(panel, style=wx.TE_MULTILINE)
        self.__text_ctrl_predict_target_sentences = wx.TextCtrl(panel, style=wx.TE_MULTILINE)
        button_pretrain = wx.Button(panel, label='Pretrain')
        button_pretrain.Bind(wx.EVT_BUTTON, self.__pretrain)
        button_train = wx.Button(panel, label='Train')
        button_train.Bind(wx.EVT_BUTTON, self.__train)
        button_predict = wx.Button(panel, label='Predict')
        button_predict.Bind(wx.EVT_BUTTON, self.__predict)
        box_train_parameters = wx.BoxSizer()
        box_train_parameters.Add(wx.StaticText(panel, label='Epochs:'), proportion=1, flag=wx.EXPAND | wx.ALL, border=5)
        box_train_parameters.Add(self.__text_ctrl_train_epochs, proportion=5, flag=wx.EXPAND | wx.ALL, border=5)
        box_train_parameters.Add(button_pretrain, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)
        box_train_parameters.Add(button_train, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)
        box_train_logs = wx.BoxSizer()
        box_train_logs.Add(self.__text_ctrl_train_logs, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)
        box_predict_button = wx.BoxSizer()
        box_predict_button.Add(button_predict, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)
        box_predict_sentences = wx.BoxSizer()
        box_predict_sentences.Add(self.__text_ctrl_predict_source_sentences, proportion=1,
                                  flag=wx.EXPAND | wx.ALL, border=5)
        box_predict_sentences.Add(self.__text_ctrl_predict_target_sentences, proportion=1,
                                  flag=wx.EXPAND | wx.ALL, border=5)
        box_main = wx.BoxSizer(wx.VERTICAL)
        box_main.Add(box_train_parameters, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)
        box_main.Add(box_train_logs, proportion=5, flag=wx.EXPAND | wx.ALL, border=5)
        box_main.Add(box_predict_button, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)
        box_main.Add(box_predict_sentences, proportion=5, flag=wx.EXPAND | wx.ALL, border=5)
        panel.SetSizer(box_main)
        self.__set_parameters()


app = wx.App()
gui = Gui()
gui.Show()
app.MainLoop()
