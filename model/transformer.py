import sys, torch, copy, random

sys.path.append("/home/neishi/workspace/my_lib")
import module.transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class LanguageModel(torch.nn.Module):
  def __init__(self, setting, tgt_lang):
    super(LanguageModel, self).__init__()
    self.tgt_lang = tgt_lang
    self.decoder =  module.transformer.TransformerDecoderBasedLanguageModel(tgt_lang.vocab_size,
                                                                            setting["decoder_vars"]["emb_dim"],
                                                                            setting["decoder_vars"]["model_dim"],
                                                                            setting["decoder_vars"]["ff_dim"],
                                                                            setting["decoder_vars"]["num_layers"],
                                                                            setting["decoder_vars"]["num_head"],
                                                                            dropout_p=setting["train_vars"]["dropout_p"],
                                                                            padding_idx=tgt_lang.vocab2index["PADDING"])
    self.generator = module.transformer.TransformerGenerator(tgt_lang.vocab_size,
                                                             setting["decoder_vars"]["model_dim"])


  def decode(self, input):
    decoder_output = self.decoder(input)
    generator_output = self.generator(decoder_output)
    return decoder_output, generator_output




class TransformerModel(torch.nn.Module):
  def __init__(self, setting, src_lang, tgt_lang):
    super(TransformerModel, self).__init__()
    self.src_lang, self.tgt_lang = src_lang, tgt_lang
    self.encoder = module.transformer.TransformerEncoder(src_lang.vocab_size,
                                                         setting["encoder_vars"]["emb_dim"],
                                                         setting["encoder_vars"]["model_dim"],
                                                         setting["encoder_vars"]["ff_dim"],
                                                         setting["encoder_vars"]["num_layers"],
                                                         setting["encoder_vars"]["num_head"],
                                                         dropout_p=setting["train_vars"]["dropout_p"],
                                                         padding_idx=src_lang.vocab2index["PADDING"])

    self.decoder = module.transformer.TransformerDecoder(tgt_lang.vocab_size,
                                                         setting["decoder_vars"]["emb_dim"],
                                                         setting["decoder_vars"]["model_dim"],
                                                         setting["decoder_vars"]["ff_dim"],
                                                         setting["decoder_vars"]["num_layers"],
                                                         setting["decoder_vars"]["num_head"],
                                                         dropout_p=setting["train_vars"]["dropout_p"],
                                                         padding_idx=src_lang.vocab2index["PADDING"])

    self.generator = module.transformer.TransformerGenerator(tgt_lang.vocab_size,
                                                             setting["decoder_vars"]["model_dim"])

    if setting["options"]["share_embedding"]:
      if src_lang.path == tgt_lang.path:
        self.decoder.embedding = self.encoder.embedding
      else:
        raise



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




  def beam_search_decode(self, decoder, para_batch, outputs, max_target_length, beam_size):
    batch_size = para_batch.tgt_batch.batch_size
    if batch_size != 1:
      raise
    outputs = outputs.expand(beam_size, -1, -1)
    masks = para_batch.src_batch.masks.expand(beam_size, -1)

    decoder_word_outputs = torch.zeros((beam_size, 1), dtype=torch.long).to(device)
    decoder_word_outputs[:, 0] = decoder.lang.vocab2index["SEQUENCE_START"] # first word is always SEQUENCE_START

    dec_state = beam_search_state(beam_size, decoder_word_outputs)

    for i in range(1, max_target_length):
      decoder_input = dec_state.decoder_word_outputs[:, :i]
      tgt_masks = (torch.zeros(decoder_input.size(), dtype=torch.long).to(device) == decoder_input)
      decoder_output = decoder(decoder_input, outputs, para_batch.src_batch.masks, tgt_masks)
      decoder_output = self.generator(decoder_output[:, -1])

      likelihood, index = decoder_output.data.topk(beam_size)

      probs = dec_state.probs.expand(-1, beam_size) + likelihood
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
          decoder_word_outputs[j, i] = decoder.lang.vocab2index["PADDING"]
        else:
          lengths[j] = dec_state.lengths[r] + 1

        if dec_state.reached_end[r] == True or index[r][c] == decoder.lang.vocab2index["SEQUENCE_END"]:
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


class beam_search_state:
  def __init__(self, beam_size, decoder_word_outputs):
    self.beam_size = beam_size
    self.att_weights = None
    self.probs = torch.zeros(self.beam_size, 1).to(device)
    self.reached_end = [False for i in range(self.beam_size)]
    self.lengths = [1 for i in range(self.beam_size)]
    self.decoder_word_outputs = decoder_word_outputs
