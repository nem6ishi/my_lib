import sys, torch, copy, random

sys.path.append("/home/neishi/workspace/my_lib")
import module.transformer
import module.rnn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Model(torch.nn.Module):
  def __init__(self, setting, src_lang, tgt_lang):
    super(Model, self).__init__()
    self.src_lang, self.tgt_lang = src_lang, tgt_lang
    self.encoder = module.transformer.TransformerEncoder(src_lang.vocab_size,
                                                         setting["encoder_vars"]["emb_dim"],
                                                         setting["encoder_vars"]["model_dim"],
                                                         setting["encoder_vars"]["ff_dim"],
                                                         setting["encoder_vars"]["num_layers"],
                                                         setting["encoder_vars"]["num_head"],
                                                         dropout_p=setting["train_vars"]["dropout_p"],
                                                         padding_idx=src_lang.vocab2index["PADDING"])

    self.attention_type = setting["decoder_vars"]["attention_type"]
    if not self.attention_type == "luong_general":
      raise
    self.decoder = module.rnn.LuongDecoderRNN(src_lang.vocab_size,
                                                     setting["decoder_vars"]["emb_dim"],
                                                     setting["decoder_vars"]["model_dim"],
                                                     setting["decoder_vars"]["num_layers"],
                                                     encoder_output_dim=setting["encoder_vars"]["model_dim"],
                                                     dropout_p=setting["train_vars"]["dropout_p"],
                                                     padding_idx=src_lang.vocab2index["PADDING"])

    if setting["options"]["share_embedding"]:
      if src_lang.path == tgt_lang.path:
        self.decoder.embedding = self.encoder.embedding
      else:
        raise



  def encode(self, input):
    return self.encoder(input)


  def decode(self, para_batch, outputs, hidden, cell, max_target_length, force_teaching_p=-1):
    batch_size = para_batch.batch_size
    flag = [False for i in range(batch_size)]

    decoder_word_outputs = torch.zeros((batch_size, max_target_length), dtype=torch.long).to(device)
    decoder_word_outputs[:, 0] = self.tgt_lang.vocab2index["SEQUENCE_START"] # first word is always SEQUENCE_START
    decoder_prob_outputs = torch.zeros((batch_size, max_target_length, self.decoder.vocab_size), dtype=torch.float).to(device)

    # for experiments of null hidden state
    if not torch.is_tensor(hidden):
      hidden = torch.zeros((self.decoder.lstm.num_layers, batch_size, self.decoder.model_dim), dtype=torch.float).to(device)
    if not torch.is_tensor(cell):
      cell = torch.zeros((self.decoder.lstm.num_layers, batch_size, self.decoder.model_dim), dtype=torch.float).to(device)

    dec_state = module.rnn.DecoderState(self.attention_type, hidden, cell)

    # decode words one by one
    for i in range(max_target_length-1):
      if force_teaching_p >= random.random():
        decoder_input = para_batch.tgt_batch.sentences[:, i].unsqueeze(1)
      else:
        decoder_input = decoder_word_outputs[:, i].unsqueeze(1)

      decoder_output = self.decoder(decoder_input, dec_state, outputs, para_batch.src_batch.masks)
      likelihood, index = decoder_output.data.topk(1)
      index = index.squeeze(1)

      for j, each in enumerate(index):
        if flag[j]:
          index[j] = self.tgt_lang.vocab2index["PADDING"]
        elif int(each) == self.tgt_lang.vocab2index["SEQUENCE_END"]:
          flag[j] = True

      decoder_word_outputs[:, i+1] = index
      decoder_prob_outputs[:, i+1] = decoder_output

      if force_teaching_p == -1 and all(flag):
        break

    return decoder_word_outputs, decoder_prob_outputs



  def beam_search_decode(self, decoder, para_batch, outputs, hidden, cell, max_target_length, beam_size):
    batch_size = para_batch.tgt_batch.batch_size
    assert batch_size == 1

    outputs = outputs.expand(beam_size, -1, -1)
    masks = para_batch.src_batch.masks.expand(beam_size, -1)

    decoder_word_outputs = torch.zeros((beam_size, 1), dtype=torch.long).to(device)
    decoder_word_outputs[:, 0] = self.tgt_lang.vocab2index["SEQUENCE_START"] # first word is always SEQUENCE_START

    # for experiments of null hidden state
    if not torch.is_tensor(hidden):
      hidden = torch.zeros((self.decoder.lstm.num_layers, batch_size, self.decoder.model_dim), dtype=torch.float).to(device)
    if not torch.is_tensor(cell):
      cell = torch.zeros((self.decoder.lstm.num_layers, batch_size, self.decoder.model_dim), dtype=torch.float).to(device)

    dec_state = beam_search_state(beam_size, self.attention_type, hidden, cell, decoder_word_outputs)

    for i in range(max_target_length-1):
      decoder_input = dec_state.decoder_word_outputs[:, i].unsqueeze(1)
      decoder_output = decoder(decoder_input, dec_state, outputs, masks)
      likelihood, index = decoder_output.data.topk(beam_size)

      tmp_probs = dec_state.probs.expand(-1, beam_size).clone()
      probs = dec_state.probs.expand(-1, beam_size) + likelihood
      for j, each in enumerate(dec_state.reached_end):
        if each:
          probs[j] = tmp_probs[j].clone()
          index[j] = self.tgt_lang.vocab2index["PADDING"]
        else:
          dec_state.lengths[j] += 1

      scores = self.calc_score(probs, dec_state.lengths, dec_state.reached_end)
      scores, sort_indexes = scores.view(beam_size*beam_size, -1).sort(0, descending=True)

      index_list, output_list = [], []
      for each in sort_indexes:
        each  = int(each)
        row, column = each//beam_size, each%beam_size
        word_output = torch.cat((decoder_input[row].unsqueeze(0), index[row][column].unsqueeze(0).unsqueeze(1)), 1).tolist()
        if word_output not in output_list:
          index_list.append((row, column))
          output_list.append(word_output)
        if len(index_list) >= beam_size:
          break

      decoder_word_outputs = torch.zeros((beam_size, i+2), dtype=torch.long).to(device)
      hidden = torch.zeros(dec_state.hidden.size()).to(device)
      cell = torch.zeros(dec_state.cell.size()).to(device)
      lengths = [0 for k in range(beam_size)]
      reached_end = [False for k in range(beam_size)]

      for j, (r, c) in enumerate(index_list):
        decoder_word_outputs[j] = torch.cat((dec_state.decoder_word_outputs[r].unsqueeze(0), index[r][c].unsqueeze(0).unsqueeze(1)), 1)
        hidden[:, j] = dec_state.hidden[:, r]
        cell[:, j] = dec_state.cell[:, r]
        dec_state.probs[j] = probs[r, c]

        if dec_state.reached_end[r] == True:
          lengths[j] = dec_state.lengths[r]
          decoder_word_outputs[j, i+1] = self.tgt_lang.vocab2index["PADDING"]
        else:
          lengths[j] = dec_state.lengths[r] + 1

        if dec_state.reached_end[r] == True or index[r][c] == self.tgt_lang.vocab2index["SEQUENCE_END"]:
          reached_end[j] = True

      dec_state.decoder_word_outputs = decoder_word_outputs
      dec_state.hidden = hidden
      dec_state.cell = cell
      dec_state.lengths = copy.deepcopy(lengths)
      dec_state.reached_end = copy.deepcopy(reached_end)

      if all(dec_state.reached_end):
        break

    return decoder_word_outputs[0].unsqueeze(0), None


  def calc_score(self, probs, pred_lengths, reached_end, alpha = 1.0):
    for i, each in enumerate(reached_end):
      if each:
        pred_lengths[i] += 1

    pred_lengths = torch.Tensor(pred_lengths).unsqueeze(1).to(device)
    length_penalty = (5+pred_lengths)**alpha / (5+1)**alpha
    scores = probs / length_penalty

    return scores



  def optimal_beam_search_decode(self, decoder, para_batch, outputs, hidden, cell, max_target_length, beam_size):
    hyperparam_r = 1.5
    ratio = 1.17

    batch_size = para_batch.tgt_batch.batch_size
    assert batch_size == 1

    best_score = -float('inf')
    outputs = outputs.expand(beam_size, -1, -1)
    masks = para_batch.src_batch.masks.expand(beam_size, -1)

    decoder_word_outputs = torch.zeros((beam_size, 1), dtype=torch.long).to(device)
    decoder_word_outputs[:, 0] = self.tgt_lang.vocab2index["SEQUENCE_START"] # first word is always SEQUENCE_START

    # for experiments of null hidden state
    if not torch.is_tensor(hidden):
      hidden = torch.zeros((self.decoder.lstm.num_layers, batch_size, self.decoder.model_dim), dtype=torch.float).to(device)
    if not torch.is_tensor(cell):
      cell = torch.zeros((self.decoder.lstm.num_layers, batch_size, self.decoder.model_dim), dtype=torch.float).to(device)

    dec_state = beam_search_state(beam_size, self.attention_type, hidden, cell, decoder_word_outputs)

    for i in range(max_target_length-1):
      decoder_input = dec_state.decoder_word_outputs[:, i].unsqueeze(1)
      decoder_output = decoder(decoder_input, dec_state, outputs, masks)
      likelihood, index = decoder_output.data.topk(beam_size)

      tmp_probs = dec_state.probs.expand(-1, beam_size).clone()
      probs = dec_state.probs.expand(-1, beam_size) + likelihood
      for j, each in enumerate(dec_state.reached_end):
        if each:
          probs[j] = tmp_probs[j].clone()
          index[j] = self.tgt_lang.vocab2index["PADDING"]
        else:
          dec_state.lengths[j] += 1


      scores = probs.clone()
      for k, each in enumerate(scores):
        scores[k] = scores[k] + hyperparam_r * min(dec_state.lengths[k], ratio*para_batch.src_batch.lengths[0])

      scores, sort_indexes = scores.view(beam_size*beam_size, -1).sort(0, descending=True)

      index_list, output_list = [], []
      for each in sort_indexes:
        each  = int(each)
        row, column = each//beam_size, each%beam_size
        word_output = torch.cat((decoder_input[row].unsqueeze(0), index[row][column].unsqueeze(0).unsqueeze(1)), 1).tolist()
        if word_output not in output_list:
          index_list.append((row, column))
          output_list.append(word_output)
        if len(index_list) >= beam_size:
          break


      decoder_word_outputs = torch.zeros((beam_size, i+2), dtype=torch.long).to(device)
      hidden = torch.zeros(dec_state.hidden.size()).to(device)
      cell = torch.zeros(dec_state.cell.size()).to(device)
      lengths = [0 for k in range(beam_size)]
      reached_end = [False for k in range(beam_size)]

      for j, (r, c) in enumerate(index_list):
        decoder_word_outputs[j] = torch.cat((dec_state.decoder_word_outputs[r].unsqueeze(0), index[r][c].unsqueeze(0).unsqueeze(1)), 1)
        hidden[:, j] = dec_state.hidden[:, r]
        cell[:, j] = dec_state.cell[:, r]
        dec_state.probs[j] = probs[r, c]
        lengths[j] = dec_state.lengths[r]

        if dec_state.reached_end[r] == True or index[r][c] == self.tgt_lang.vocab2index["SEQUENCE_END"]:
          reached_end[j] = True
          if best_score < scores[j]:
            best_score = scores[j].clone()
            best_word_output = decoder_word_outputs[j].clone()

      dec_state.decoder_word_outputs = decoder_word_outputs
      dec_state.hidden = hidden
      dec_state.cell = cell
      dec_state.lengths = copy.deepcopy(lengths)
      dec_state.reached_end = copy.deepcopy(reached_end)

      if all(dec_state.reached_end):
        break
      elif best_score >= dec_state.probs[0] + hyperparam_r * ratio*para_batch.src_batch.lengths[0]:
          #print(best_score, dec_state.probs[0] + hyperparam_r * ratio*para_batch.src_batch.lengths[0], dec_state.probs[0], hyperparam_r, ratio, para_batch.src_batch.lengths[0], flush=True)
          return best_word_output.unsqueeze(0), None

    return decoder_word_outputs[0].unsqueeze(0), None


class beam_search_state:
  def __init__(self, beam_size, attention_type, hidden, cell, decoder_word_outputs):
    self.beam_size = beam_size

    self.hidden = hidden.expand(-1, self.beam_size, -1).contiguous()
    self.cell = cell.expand(-1, self.beam_size, -1).contiguous()
    if attention_type == "luong_general":
      size = (self.hidden.size(1), self.hidden.size(2))
      self.h_tilde = torch.zeros(size).to(device)
    self.att_weights = None

    self.probs = torch.zeros(self.beam_size, 1).to(device)
    self.reached_end = [False for i in range(self.beam_size)]
    self.lengths = [1 for i in range(self.beam_size)]

    self.decoder_word_outputs = decoder_word_outputs
