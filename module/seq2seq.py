import torch, copy
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class EncoderRNN(torch.nn.Module):
  def __init__(self, vocab_size, emb_dim, model_dim, num_layers, bi_directional, dropout_p, rnn_type="lstm", padding_idx=0, reverse_input=False):
    super(EncoderRNN, self).__init__()

    self.reverse_input = reverse_input

    self.vocab_size = vocab_size
    self.emb_dim = emb_dim
    self.model_dim = model_dim
    self.num_layers = num_layers
    self.bi_directional = bi_directional
    self.n_directions = 2 if self.bi_directional else 1

    self.dropout_p = dropout_p
    self.rnn_type = rnn_type

    self.embedding = torch.nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=padding_idx)
    if self.rnn_type == "lstm":
        self.rnn = torch.nn.LSTM(self.emb_dim, self.model_dim, self.num_layers, batch_first=True, dropout=self.dropout_p, bidirectional=self.bi_directional)
    elif self.rnn_type == "gru":
        self.rnn = torch.nn.GRU(self.emb_dim, self.model_dim, self.num_layers, batch_first=True, dropout=self.dropout_p, bidirectional=self.bi_directional)
    else:
        raise
    self.dropout = torch.nn.Dropout(self.dropout_p)


  def forward(self, input):
    input = input.copy()
    length = ###
    input, sort_indexes = self.sort(input, length)

    if self.reverse_input:
      input = self.reverse_sentence(input)

    embedded = self.dropout(self.embedding(input))
    embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, length, batch_first=True) # form matrix into list, to care paddings in rnn
    outputs, last_hidden = self.rnn(embedded)
    outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True) # reform into matrix

    outputs, last_hidden = self.resort(sort_indexes, outputs, last_hidden)

    return outputs, last_hidden


  def sort(self, input, length):
    length, sort_indexes = torch.LongTensor(length).sort(0, descending=True)
    length = length.tolist()
    input = input[sort_indexes]
    return input, sort_indexes


  def resort(self, sort_indexes, outputs, last_hidden):
    _, resort_indexes = sort_indexes.sort(0, descending=False)

    outputs = outputs[resort_indexes]
    if rnn_type=="lstm":
      hidden = last_hidden[0][:, resort_indexes]
      cell = last_hidden[1][:, resort_indexes]
      last_hidden = (hidden, cell)
    elif self.rnn_type == "gru":
      last_hidden = last_hidden[:, resort_indexes]

    return outputs, last_hidden



  def reverse_sentence(self, input, length):
    batch_length = input.size(1)
    reversed_input = torch.LongTensor(input.size()).to(device) # keep original sentences from reversing

    # turn all
    reversed_input = input[:, list(range(batch_length-1, -1, -1))]

    # shift words and paddings
    for i, each in enumerate(length):
      num_padding = batch_length - each
      if num_padding > 0:
        reversed_input[i] = torch.cat((reversed_input[i, num_padding:], reversed_input[i, :num_padding]), 0)

    return reversed_input



def linear_transform_for_decoder_hidden(self, hidden, cell):
  if not self.need_transform:
    return (hidden, cell)

  batch_size = hidden.size(1)

  hidden = hidden.transpose(0, 1).contiguous().view(batch_size, -1)
  hidden = torch.tanh(self.linear_hidden(hidden))
  hidden = hidden.view(batch_size, self.decoder_num_layers, self.decoder_model_dim).transpose(0, 1).contiguous()

  cell = cell.transpose(0, 1).contiguous().view(batch_size, -1)
  cell = torch.tanh(self.linear_cell(cell))
  cell = cell.view(batch_size, self.decoder_num_layers, self.decoder_model_dim).transpose(0, 1).contiguous()

  return (hidden, cell)




class BahdanauDecoderRNN(torch.nn.Module):
  def __init__(self, setting, lang):
    super(BahdanauDecoderRNN, self).__init__()

    self.lang = lang

    self.emb_dim = setting["decoder_vars"]["emb_dim"]
    self.model_dim = setting["decoder_vars"]["model_dim"]
    self.num_layers = setting["decoder_vars"]["num_layers"]

    self.dropout_p = setting["train_vars"]["dropout_p"]
    self.encoder_n_directions = 2 if setting["encoder_vars"]["bi_directional"] else 1
    self.encoder_model_dim = setting["encoder_vars"]["model_dim"]

    self.embedding = torch.nn.Embedding(self.lang.vocab_size, self.emb_dim, padding_idx=self.lang.vocab2index["PADDING"])
    if setting["paths"]["tgt_vocab_emb"] != None:
      self.embedding.weight.data = torch.from_numpy(np.load(setting["paths"]["tgt_vocab_emb"])).float()

    self.dropout = torch.nn.Dropout(self.dropout_p)
    self.lstm = torch.nn.LSTM(self.emb_dim + self.encoder_model_dim*self.encoder_n_directions, self.model_dim, num_layers=self.num_layers, batch_first=True, dropout=self.dropout_p)
    self.out = torch.nn.Linear(self.model_dim, self.lang.vocab_size)

    self.attn = torch.nn.Linear(self.num_layers*self.model_dim + self.encoder_model_dim*self.encoder_n_directions, self.model_dim)
    self.v = torch.nn.Linear(self.model_dim, 1, bias=False)


  def forward(self, input, dec_state, encoder_outputs, src_mask):
    embedded = self.dropout(self.embedding(input))

    contexts, dec_state.att_weights = self.attention(dec_state.hidden, encoder_outputs, src_mask)

    lstm_input = torch.cat((embedded, contexts), 2)
    output, (dec_state.hidden, dec_state.cell) = self.lstm(lstm_input, (dec_state.hidden, dec_state.cell))

    output = self.out(output.squeeze(1))
    output = torch.nn.functional.log_softmax(output, dim=1).squeeze(1)

    return output


  def attention(self, hidden, encoder_outputs, src_mask):
    batch_size = encoder_outputs.size(0)
    input_sentence_length = encoder_outputs.size(1)

    H = hidden.transpose(0, 1).contiguous().view(batch_size, self.num_layers*self.model_dim)
    H = H.unsqueeze(1).expand(H.size(0), input_sentence_length, H.size(1))

    energy = torch.tanh(self.attn(torch.cat((H, encoder_outputs), 2)))
    energy = self.v(energy).transpose(1,2)

    energy.data.masked_fill_(src_mask.unsqueeze(1), float('-inf'))

    att_weights = torch.nn.functional.softmax(energy, dim=2)
    contexts = torch.bmm(att_weights, encoder_outputs)

    return contexts, att_weights



class LuongDecoderRNN(torch.nn.Module):
  def __init__(self, setting, lang):
    super(LuongDecoderRNN, self).__init__()

    self.lang = lang

    self.emb_dim = setting["decoder_vars"]["emb_dim"]
    self.model_dim = setting["decoder_vars"]["model_dim"]
    self.num_layers = setting["decoder_vars"]["num_layers"]
    self.dropout_p = setting["train_vars"]["dropout_p"]

    self.encoder_n_directions = 2 if setting["encoder_vars"]["bi_directional"] else 1
    self.encoder_model_dim = setting["encoder_vars"]["model_dim"]

    self.embedding = torch.nn.Embedding(self.lang.vocab_size, self.emb_dim, padding_idx=self.lang.vocab2index["PADDING"])
    if setting["paths"]["tgt_vocab_emb"] != None:
      self.embedding.weight.data = torch.from_numpy(np.load(setting["paths"]["tgt_vocab_emb"])).float()

    self.dropout = torch.nn.Dropout(self.dropout_p)
    self.lstm = torch.nn.LSTM(self.embedding.embedding_dim+self.model_dim, self.model_dim, num_layers=self.num_layers, batch_first=True, dropout=self.dropout_p)
    self.Wa = torch.nn.Linear(self.encoder_n_directions*self.encoder_model_dim, self.num_layers*self.model_dim, bias=False)
    self.Wc = torch.nn.Linear(self.num_layers*self.model_dim + self.encoder_n_directions*self.encoder_model_dim, self.model_dim, bias=False)
    self.Ws = torch.nn.Linear(self.model_dim, self.lang.vocab_size)


  def forward(self, input, dec_state, encoder_outputs, src_mask):
    batch_size = input.size(0)

    embedded = self.dropout(self.embedding(input))
    lstm_input = torch.cat((embedded, dec_state.h_tilde.unsqueeze(1)), 2) # input feeding

    output, (dec_state.hidden, dec_state.cell) = self.lstm(lstm_input, (dec_state.hidden, dec_state.cell))

    contexts, dec_state.att_weights = self.attention(dec_state.hidden, encoder_outputs, src_mask)

    dec_state.h_tilde = torch.tanh(self.Wc(torch.cat((dec_state.hidden.transpose(0, 1).contiguous().view(batch_size, -1), contexts), 1)))
    output = self.Ws(dec_state.h_tilde)
    output = torch.nn.functional.log_softmax(output, dim=1).squeeze(1)

    return output


  def attention(self, hidden, encoder_outputs, mask):
    if not torch.is_tensor(encoder_outputs): # not use attention
      contexts = torch.zeros(batch_size, self.encoder_model_dim*self.encoder_n_directions).to(device)
      return contexts, None

    batch_size = encoder_outputs.size(0)
    input_sentence_length = encoder_outputs.size(1)

    hidden = hidden.transpose(0, 1).contiguous().view(batch_size, -1)
    score = self.Wa(encoder_outputs)
    score = torch.bmm(hidden.unsqueeze(1), score.transpose(1,2)).squeeze(1)
    score.data.masked_fill_(mask, float('-inf'))

    att_weights = torch.nn.functional.softmax(score, dim=1)
    contexts = torch.bmm(att_weights.unsqueeze(1), encoder_outputs).squeeze(1)

    return contexts, att_weights



class DecoderRNN(torch.nn.Module):
  def __init__(self, setting, lang):
    super(DecoderRNN, self).__init__()

    self.lang = lang

    self.emb_dim = setting["decoder_vars"]["emb_dim"]
    self.model_dim = setting["decoder_vars"]["model_dim"]
    self.num_layers = setting["decoder_vars"]["num_layers"]
    self.dropout_p = setting["train_vars"]["dropout_p"]

    self.embedding = torch.nn.Embedding(self.lang.vocab_size, self.emb_dim, padding_idx=self.lang.vocab2index["PADDING"])
    if setting["paths"]["tgt_vocab_emb"] != None:
      self.embedding.weight.data = torch.from_numpy(np.load(setting["paths"]["tgt_vocab_emb"])).float()

    self.dropout = torch.nn.Dropout(self.dropout_p)
    self.lstm = torch.nn.LSTM(self.emb_dim, self.model_dim, num_layers=self.num_layers, batch_first=True, dropout=self.dropout_p)
    self.out = torch.nn.Linear(self.model_dim, self.lang.vocab_size)


  def forward(self, input, dec_state, _encoder_outputs, _src_mask):
    embedded = self.dropout(self.embedding(input))

    output, (dec_state.hidden, dec_state.cell) = self.lstm(embedded, (dec_state.hidden, dec_state.cell)) # SLOW

    output = self.out(output.squeeze(1))
    output = torch.nn.functional.log_softmax(output, dim=1).squeeze(1)

    return output



class DecoderState:
  def __init__(self, attention_type, hidden, cell=None):
    self.hidden = hidden
    self.cell = cell
    if attention_type == "luong_general":
      size = (hidden.size(1), hidden.size(2))
      self.h_tilde = torch.zeros(size).to(device)
    self.att_weights = None
