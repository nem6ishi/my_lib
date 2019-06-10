import sys, torch, copy, random

sys.path.append("/home/neishi/workspace/my_lib")
import module.rnn_transformer
import module.transformer
import model.transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class RNNTransformerModel(torch.nn.Module):
  def __init__(self, setting, src_lang, tgt_lang):
    super(RNNTransformerModel, self).__init__()
    self.src_lang, self.tgt_lang = src_lang, tgt_lang
    self.encoder = module.rnn_transformer.RNNTransformerEncoder(src_lang.vocab_size,
                                                         setting["encoder_vars"]["emb_dim"],
                                                         setting["encoder_vars"]["model_dim"],
                                                         setting["encoder_vars"]["ff_dim"],
                                                         setting["encoder_vars"]["num_layers"],
                                                         setting["encoder_vars"]["num_head"],
                                                         setting["encoder_vars"]["bi_directional"],
                                                         dropout_p=setting["train_vars"]["dropout_p"],
                                                         padding_idx=src_lang.vocab2index["PADDING"])

    self.decoder = module.rnn_transformer.RNNTransformerDecoder(tgt_lang.vocab_size,
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
    decoder_output, _hidden = self.decoder(tgt_sent, encoder_outputs, src_mask)
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
    hidden = None

    layer_cache = {}
    for i in range(self.decoder.num_layers):
      layer_cache[i] = {"kv_mask": None, "key": None, "value": None}

    # decode words one by one
    for i in range(1, max_target_length):
      decoder_output, hidden = self.decoder(decoder_word_outputs[:, i-1:i], encoder_outputs, src_mask, layer_cache, hidden)
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
