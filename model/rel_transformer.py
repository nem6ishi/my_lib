import sys, torch, copy, random

sys.path.append("/home/neishi/workspace/my_lib")
import module.rel_transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class RelTransformerModel(torch.nn.Module):
  def __init__(self, setting, src_lang, tgt_lang):
    super(RelTransformerModel, self).__init__()
    self.src_lang, self.tgt_lang = src_lang, tgt_lang
    self.encoder = module.rel_transformer.TransformerEncoder(src_lang.vocab_size,
                                                         setting["encoder_vars"]["emb_dim"],
                                                         setting["encoder_vars"]["model_dim"],
                                                         setting["encoder_vars"]["ff_dim"],
                                                         setting["encoder_vars"]["num_layers"],
                                                         setting["encoder_vars"]["num_head"],
                                                         dropout_p=setting["train_vars"]["dropout_p"],
                                                         padding_idx=src_lang.vocab2index["PADDING"])

    self.decoder = module.rel_transformer.TransformerDecoder(tgt_lang.vocab_size,
                                                         setting["decoder_vars"]["emb_dim"],
                                                         setting["decoder_vars"]["model_dim"],
                                                         setting["decoder_vars"]["ff_dim"],
                                                         setting["decoder_vars"]["num_layers"],
                                                         setting["decoder_vars"]["num_head"],
                                                         dropout_p=setting["train_vars"]["dropout_p"],
                                                         padding_idx=src_lang.vocab2index["PADDING"])

    self.generator = module.rel_transformer.TransformerGenerator(tgt_lang.vocab_size,
                                                             setting["decoder_vars"]["model_dim"])

    if setting["options"]["share_embedding"]:
      if src_lang.path == tgt_lang.path:
        self.decoder.embedding = self.encoder.embedding
      else:
        raise


  def translate_for_train(self, batch):
    outputs = self.encoder(batch.src_batch.sentences)
    prob_outputs = self.decode_for_train(batch.tgt_batch.sentences[:, :-1],
                                            outputs,
                                            batch.src_batch.masks)
    return prob_outputs



  def translate(self, batch, max_length, reverse_output=False):
    outputs = self.encoder(batch.src_batch.sentences)
    word_outputs, prob_outputs = self.decode(outputs,
                                             batch.src_batch.masks,
                                             max_length,
                                             reverse_output)
    return word_outputs, prob_outputs


  def encode(self, input):
    return self.encoder(input)



  def decode_for_train(self, tgt_sent, encoder_outputs, src_mask):
    decoder_output = self.decoder(tgt_sent, encoder_outputs, src_mask)
    decoder_prob_outputs = self.generator(decoder_output)
    return decoder_prob_outputs



  def decode(self, encoder_outputs, src_mask, max_target_length, reverse_output=False):

    start_token = self.tgt_lang.vocab2index["SEQUENCE_START"]
    end_token = self.tgt_lang.vocab2index["SEQUENCE_END"]
    if reverse_output:
      start_token, end_token = end_token, start_token

    batch_size = encoder_outputs.size(0)
    flag = [False for i in range(batch_size)]
    decoder_word_outputs = torch.zeros((batch_size, max_target_length), dtype=torch.long).to(device)
    decoder_word_outputs[:, 0] = start_token # first word
    decoder_prob_outputs = torch.zeros((batch_size, max_target_length, self.tgt_lang.vocab_size), dtype=torch.float).to(device)

    layer_cache = {}
    for i in range(self.decoder.num_layers):
      layer_cache[i] = {"kv_mask": None, "key": None, "value": None}

    # decode words one by one
    for i in range(1, max_target_length):
      decoder_output = self.decoder(decoder_word_outputs[:, i-1:i], encoder_outputs, src_mask, i-1, layer_cache)
      generator_output = self.generator(decoder_output[:, -1])

      likelihood, index = generator_output.data.topk(1)
      index = index.squeeze(1)

      for j, each in enumerate(index):
        if flag[j]:
          index[j] = self.tgt_lang.vocab2index["PADDING"]
        elif int(each) == end_token:
          flag[j] = True

      decoder_word_outputs[:, i] = index
      decoder_prob_outputs[:, i] = generator_output

      if all(flag):
        break

    return decoder_word_outputs, decoder_prob_outputs




  def beam_search_decode(self, outputs, src_mask, max_target_length, beam_size):
    batch_size = outputs.size(0)
    assert batch_size == 1
    outputs = outputs.expand(beam_size, -1, -1)
    masks = src_mask.expand(beam_size, -1)

    decoder_word_outputs = torch.zeros((beam_size, 1), dtype=torch.long).to(device)
    decoder_word_outputs[:, 0] = self.tgt_lang.vocab2index["SEQUENCE_START"] # first word is always SEQUENCE_START

    dec_state = beam_search_state(beam_size, decoder_word_outputs)

    for i in range(1, max_target_length):
      decoder_input = dec_state.decoder_word_outputs[:, :i]
      tgt_masks = (torch.zeros(decoder_input.size(), dtype=torch.long).to(device) == decoder_input)
      decoder_output = self.decoder(decoder_input, outputs, masks)
      decoder_output = self.generator(decoder_output[:, -1])

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

      decoder_word_outputs = torch.zeros((beam_size, i+1), dtype=torch.long).to(device)
      lengths = [0 for k in range(beam_size)]
      reached_end = [False for k in range(beam_size)]

      for j, (r, c) in enumerate(index_list):
        decoder_word_outputs[j] = torch.cat((dec_state.decoder_word_outputs[r], index[r][c].unsqueeze(0)), 0)
        dec_state.probs[j] = probs[r, c]

        if dec_state.reached_end[r] == True:
          lengths[j] = dec_state.lengths[r]
          decoder_word_outputs[j, i] = self.tgt_lang.vocab2index["PADDING"]
        else:
          lengths[j] = dec_state.lengths[r] + 1

        if dec_state.reached_end[r] == True or index[r][c] == self.tgt_lang.vocab2index["SEQUENCE_END"]:
          reached_end[j] = True

      dec_state.decoder_word_outputs = decoder_word_outputs
      dec_state.lengths = copy.deepcopy(lengths)
      dec_state.reached_end = copy.deepcopy(reached_end)

      if all(dec_state.reached_end):
        break

    return decoder_word_outputs[0].unsqueeze(0), None


  def calc_score(self, probs, pred_lengths, reached_end, alpha=1.0):
    for i, each in enumerate(reached_end):
      if each:
        pred_lengths[i] += 1

    pred_lengths = torch.Tensor(pred_lengths).unsqueeze(1).to(device)
    length_penalty = (5+pred_lengths)**alpha / (5+1)**alpha
    scores = probs / length_penalty

    return scores



  def optimal_beam_search_decode(self, outputs, src_mask, max_target_length, beam_size):
    hyperparam_r = 1.6
    ratio = 1.17 # from aspec sp16k dev

    src_sent_len = float(src_mask.size(1))

    batch_size = outputs.size(0)
    assert batch_size == 1

    best_score = -float('inf')
    outputs = outputs.expand(beam_size, -1, -1)
    masks = src_mask.expand(beam_size, -1)

    decoder_word_outputs = torch.zeros((beam_size, 1), dtype=torch.long).to(device)
    decoder_word_outputs[:, 0] = self.tgt_lang.vocab2index["SEQUENCE_START"] # first word is always SEQUENCE_START

    dec_state = beam_search_state(beam_size, decoder_word_outputs)

    for i in range(1, max_target_length):
      decoder_input = dec_state.decoder_word_outputs[:, :i]
      tgt_masks = (torch.zeros(decoder_input.size(), dtype=torch.long).to(device) == decoder_input)
      decoder_output = self.decoder(decoder_input, outputs, masks)
      decoder_output = self.generator(decoder_output[:, -1])

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
        scores[k] = scores[k] + hyperparam_r * float(min(dec_state.lengths[k], ratio*src_sent_len))
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

      decoder_word_outputs = torch.zeros((beam_size, i+1), dtype=torch.long).to(device)
      lengths = [0 for k in range(beam_size)]
      reached_end = [False for k in range(beam_size)]

      for j, (r, c) in enumerate(index_list):
        decoder_word_outputs[j] = torch.cat((dec_state.decoder_word_outputs[r], index[r][c].unsqueeze(0)), 0)
        dec_state.probs[j] = probs[r, c]

        if dec_state.reached_end[r] == True or index[r][c] == self.tgt_lang.vocab2index["SEQUENCE_END"]:
          reached_end[j] = True
          if best_score < scores[j]:
            best_score = scores[j].clone()
            best_word_output = decoder_word_outputs[j].clone()

      dec_state.decoder_word_outputs = decoder_word_outputs
      dec_state.lengths = copy.deepcopy(lengths)
      dec_state.reached_end = copy.deepcopy(reached_end)

      if all(dec_state.reached_end):
        break
      elif best_score >= dec_state.probs[0] + hyperparam_r * ratio*src_sent_len:
          #print(best_score, dec_state.probs[0] + hyperparam_r * ratio*src_sent_len, dec_state.probs[0], hyperparam_r, ratio, src_sent_len, flush=True)
          return best_word_output.unsqueeze(0), None

    return decoder_word_outputs[0].unsqueeze(0), None




class beam_search_state:
  def __init__(self, beam_size, decoder_word_outputs):
    self.beam_size = beam_size
    self.att_weights = None
    self.probs = torch.zeros(self.beam_size, 1).to(device)
    self.reached_end = [False for i in range(self.beam_size)]
    self.lengths = [1 for i in range(self.beam_size)]
    self.decoder_word_outputs = decoder_word_outputs





"""
class BidirectionalEncoderTransformerModel(TransformerModel):
  def __init__(self, setting, src_lang, tgt_lang):
    super(BidirectionalEncoderTransformerModel, self).__init__(setting, src_lang, tgt_lang)
    self.src_lang, self.tgt_lang = src_lang, tgt_lang
    self.encoder = module.rel_transformer.BidirectionalTransformerEncoder(src_lang.vocab_size,
                                                         setting["encoder_vars"]["emb_dim"],
                                                         setting["encoder_vars"]["model_dim"],
                                                         setting["encoder_vars"]["ff_dim"],
                                                         setting["encoder_vars"]["num_layers"],
                                                         setting["encoder_vars"]["num_head"],
                                                         dropout_p=setting["train_vars"]["dropout_p"],
                                                         padding_idx=src_lang.vocab2index["PADDING"])

    self.decoder = module.rel_transformer.TransformerDecoder(tgt_lang.vocab_size,
                                                         setting["decoder_vars"]["emb_dim"],
                                                         setting["decoder_vars"]["model_dim"],
                                                         setting["decoder_vars"]["ff_dim"],
                                                         setting["decoder_vars"]["num_layers"],
                                                         setting["decoder_vars"]["num_head"],
                                                         dropout_p=setting["train_vars"]["dropout_p"],
                                                         padding_idx=src_lang.vocab2index["PADDING"])

    self.generator = module.rel_transformer.TransformerGenerator(tgt_lang.vocab_size,
                                                             setting["decoder_vars"]["model_dim"])

    if setting["options"]["share_embedding"]:
      if src_lang.path == tgt_lang.path:
        self.decoder.embedding = self.encoder.embedding
      else:
        raise




class FixedLengthTransformerLanguageModel(torch.nn.Module):
  def __init__(self, setting, tgt_lang):
    super(FixedLengthTransformerLanguageModel, self).__init__()
    self.tgt_lang = tgt_lang
    self.decoder =  module.rel_transformer.TransformerDecoderBasedLanguageModel(tgt_lang.vocab_size,
                                                                            setting["model_vars"]["emb_dim"],
                                                                            setting["model_vars"]["model_dim"],
                                                                            setting["model_vars"]["ff_dim"],
                                                                            setting["model_vars"]["num_layers"],
                                                                            setting["model_vars"]["num_head"],
                                                                            dropout_p=setting["train_vars"]["dropout_p"],
                                                                            padding_idx=tgt_lang.vocab2index["PADDING"])
    self.generator = module.rel_transformer.TransformerGenerator(tgt_lang.vocab_size,
                                                             setting["model_vars"]["model_dim"])
    self.num_gram = setting["model_vars"]["num_gram"]


  def predict_for_training(self, input):
    batch_size = input.size(0)
    length = input.size(1)

    prob_outputs = torch.zeros((batch_size, length, self.tgt_lang.vocab_size), dtype=torch.float).to(device)

    #backward
    reverse_idx = list(range(length))
    reverse_idx.reverse()
    input = input[:, reverse_idx]

    for i in range(length):
      if i==0:
        input_for_lang_model = torch.zeros((batch_size, self.num_gram), dtype=torch.long).to(device)
      elif i < self.num_gram:
        padding = torch.zeros((batch_size, self.num_gram-i), dtype=torch.long).to(device)
        input_slice = input[:, :i]
        input_for_lang_model = torch.cat((padding, input_slice), 1)
      else:
        input_for_lang_model = input[:, i-self.num_gram:i]

      decoder_output = self.decoder(input_for_lang_model)
      prob_output = self.generator(decoder_output[:, -1])
      prob_outputs[:, i] = prob_output

    prob_outputs = prob_outputs[:, reverse_idx] #backward

    return prob_outputs
"""
