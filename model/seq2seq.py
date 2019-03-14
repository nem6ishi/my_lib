import sys, torch, copy, random

sys.path.append("/home/neishi/workspace/my_lib")
import util.corpus, util.language
import module.rnn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Seq2SeqModel(torch.nn.Module):
  def __init__(self, setting, src_lang, tgt_lang):
    super(Seq2SeqModel, self).__init__()

    self.encoder = module.rnn.OldEncoderRNN(setting, src_lang)

    self.attention_type = setting["decoder_vars"]["attention_type"]
    if self.attention_type == "luong_general":
      self.decoder = module.rnn.OldLuongDecoderRNN(setting, tgt_lang)
    else:
      raise

    if setting["options"]["share_embedding"]:
      assert setting["paths"]["src_vocab"] == setting["paths"]["tgt_vocab"]
      self.decoder.embedding = self.encoder.embedding



  def encode(self, batch):
    outputs, (hidden, cell) = self.encoder(batch)
    return outputs, (hidden, cell)



  def decode(self, decoder, para_batch, outputs, hidden, cell, max_target_length, force_teaching_p=-1):
    batch_size = para_batch.batch_size
    flag = [False for i in range(batch_size)]

    decoder_word_outputs = torch.zeros((batch_size, max_target_length), dtype=torch.long).to(device)
    decoder_word_outputs[:, 0] = decoder.lang.vocab2index["SEQUENCE_START"] # first word is always SEQUENCE_START
    decoder_prob_outputs = torch.zeros((batch_size, max_target_length, decoder.lang.vocab_size), dtype=torch.float).to(device)

    # for experiments of null hidden state
    if not torch.is_tensor(hidden):
      hidden = torch.zeros((decoder.lstm.num_layers, batch_size, decoder.hidden_size), dtype=torch.float).to(device)
    if not torch.is_tensor(cell):
      cell = torch.zeros((decoder.lstm.num_layers, batch_size, decoder.hidden_size), dtype=torch.float).to(device)

    dec_state = module.DecoderState(self.attention_type, hidden, cell)

    # decode words one by one
    for i in range(max_target_length-1):
      if force_teaching_p >= random.random():
        decoder_input = para_batch.tgt_batch.sentences[:, i].unsqueeze(1)
      else:
        decoder_input = decoder_word_outputs[:, i].unsqueeze(1)

      decoder_output = decoder(decoder_input, dec_state, outputs, para_batch.src_batch.masks)
      likelihood, index = decoder_output.data.topk(1)
      index = index.squeeze(1)

      for j, each in enumerate(index):
        if flag[j]:
          index[j] = decoder.lang.vocab2index["PADDING"]
        elif int(each) == decoder.lang.vocab2index["SEQUENCE_END"]:
          flag[j] = True

      decoder_word_outputs[:, i+1] = index
      decoder_prob_outputs[:, i+1] = decoder_output

      if force_teaching_p == -1 and all(flag):
        break

    return decoder_word_outputs, decoder_prob_outputs
