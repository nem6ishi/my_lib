import sys, torch, copy, random
from logging import getLogger
logger = getLogger(__name__)

sys.path.append("/home/neishi/workspace/my_lib")
import module.deep_fusion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class DeepFusionModel(torch.nn.Module):
  def __init__(self, setting, src_lang, tgt_lang, t_model, l_model, num_pred_more=0):
    super(DeepFusionModel, self).__init__()
    self.src_lang = src_lang
    self.tgt_lang = tgt_lang
    self.t_model = t_model
    self.l_model = l_model
    self.deep_fusion = module.deep_fusion.DeepFusionModule(tgt_lang.vocab_size,
                                                           self.t_model.decoder.model_dim,
                                                           self.l_model.decoder.model_dim,
                                                           setting["model_vars"]["model_dim"],
                                                           dropout_p=setting["train_vars"]["dropout_p"])
    self.num_pred_more = num_pred_more


  def translate(self, input, max_target_length, tgt_sent=None):
    src_mask = (input==torch.zeros(input.size(), dtype=torch.long).to(device))
    encoder_outputs = self.t_model.encode(input)


    start_token = self.tgt_lang.vocab2index["SEQUENCE_START"]
    end_token = self.tgt_lang.vocab2index["SEQUENCE_END"]

    batch_size = input.size(0)
    flag = [False for i in range(batch_size)]
    decoder_word_outputs = torch.zeros((batch_size, max_target_length), dtype=torch.long).to(device)
    decoder_word_outputs[:, 0] = start_token
    decoder_prob_outputs = torch.zeros((batch_size, max_target_length, self.tgt_lang.vocab_size), dtype=torch.float).to(device)

    layer_cache = {}
    for i in range(self.t_model.decoder.num_layers):
      layer_cache[i] = {"kv_mask": None, "key": None, "value": None}

    # decode words one by one
    for i in range(1, max_target_length):

      ### pseudo predict loop start ###
      if self.num_pred_more > 0:
        pseudo_flag = copy.deepcopy(flag)
        pseudo_word_outputs = torch.zeros((batch_size, self.num_pred_more+1), dtype=torch.long).to(device)
        pseudo_prob_outputs = torch.zeros((batch_size, self.num_pred_more+1, self.tgt_lang.vocab_size), dtype=torch.float).to(device)
        tmp_layer_cache = copy.deepcopy(layer_cache)

        for k in range(self.num_pred_more+1):
          # previous_word
          if k==0:
            if not torch.is_tensor(tgt_sent):
              decoder_input = decoder_word_outputs[:, i-1:i]
            else:
              decoder_input = tgt_sent[:, i-1:i]
          else:
            decoder_input = pseudo_word_outputs[:, k-1:k]
          decoder_output = self.t_model.decoder(decoder_input, encoder_outputs, src_mask, time_step=i-1+k, layer_cache=tmp_layer_cache)
          generator_output = self.t_model.generator(decoder_output[:, -1])

          likelihood, index = generator_output.data.topk(1)
          index = index.squeeze(1)

          for j, each in enumerate(index):
            if pseudo_flag[j]:
              index[j] = self.tgt_lang.vocab2index["PADDING"]
            elif int(each) == end_token:
              pseudo_flag[j] = True

          pseudo_word_outputs[:, k] = index
          pseudo_prob_outputs[:, k] = generator_output

          if k==0:
            t_out = decoder_output[:, -1]
            layer_cache = copy.deepcopy(tmp_layer_cache)
        ### pseudo predict loop end ###

        #language model
        sort_index = torch.tensor(list(range(pseudo_word_outputs.size(1)-1, -1, -1)))
        inputs = pseudo_word_outputs[:, sort_index]
        lang_model_decoder_output, lang_model_generator_output = self.l_model.decode(inputs[:, :-1])
        l_out = lang_model_decoder_output[:, -1]

        t_out.requires_grad_()
        l_out.requires_grad_()
        deep_fusion_output = self.deep_fusion(t_out, l_out)

      else:
        decoder_output = self.t_model.decoder(decoder_word_outputs[:, i-1:i], encoder_outputs, src_mask, time_step=i-1, layer_cache=layer_cache)
        generator_output = self.t_model.generator(decoder_output[:, -1])
        deep_fusion_output = generator_output

      likelihood, index = deep_fusion_output.data.topk(1)
      index = index.squeeze(1)

      for j, each in enumerate(index):
        if flag[j]:
          index[j] = self.tgt_lang.vocab2index["PADDING"]
        elif int(each) == end_token:
          flag[j] = True

      decoder_word_outputs[:, i] = index
      decoder_prob_outputs[:, i] = deep_fusion_output

      if all(flag):
        break

    return decoder_prob_outputs, decoder_word_outputs





  def translate_forward(self, input, max_target_length, tgt_sent=None):
    src_mask = (input==torch.zeros(input.size(), dtype=torch.long).to(device))
    encoder_outputs = self.t_model.encode(input)


    start_token = self.tgt_lang.vocab2index["SEQUENCE_START"]
    end_token = self.tgt_lang.vocab2index["SEQUENCE_END"]

    batch_size = input.size(0)
    flag = [False for i in range(batch_size)]
    decoder_word_outputs = torch.zeros((batch_size, max_target_length), dtype=torch.long).to(device)
    decoder_word_outputs[:, 0] = start_token
    decoder_prob_outputs = torch.zeros((batch_size, max_target_length, self.tgt_lang.vocab_size), dtype=torch.float).to(device)

    layer_cache = {}
    for i in range(self.t_model.decoder.num_layers):
      layer_cache[i] = {"kv_mask": None, "key": None, "value": None}

    # decode words one by one
    for i in range(1, max_target_length):

      if not torch.is_tensor(tgt_sent):
        decoder_input = decoder_word_outputs[:, i-1:i]
      else:
        decoder_input = tgt_sent[:, i-1:i]
      decoder_output = self.t_model.decoder(decoder_input, encoder_outputs, src_mask, time_step=i-1, layer_cache=layer_cache)
      t_out = decoder_output[:, -1]

      #language model
      if not torch.is_tensor(tgt_sent):
        decoder_input = decoder_word_outputs[:, :i]
      else:
        decoder_input = tgt_sent[:, :i]

      lang_model_decoder_output, lang_model_generator_output = self.l_model.decode(decoder_input)
      l_out = lang_model_decoder_output[:, -1]

      t_out.requires_grad_()
      l_out.requires_grad_()
      deep_fusion_output = self.deep_fusion(t_out, l_out)


      likelihood, index = deep_fusion_output.data.topk(1)
      index = index.squeeze(1)

      for j, each in enumerate(index):
        if flag[j]:
          index[j] = self.tgt_lang.vocab2index["PADDING"]
        elif int(each) == end_token:
          flag[j] = True

      decoder_word_outputs[:, i] = index
      decoder_prob_outputs[:, i] = deep_fusion_output

      if all(flag):
        break

    return decoder_prob_outputs, decoder_word_outputs


  def translate_forward_train(self, input, tgt_sent):
    src_mask = (input==torch.zeros(input.size(), dtype=torch.long).to(device))
    encoder_outputs = self.t_model.encode(input)

    decoder_input = tgt_sent
    decoder_output = self.t_model.decoder(tgt_sent, encoder_outputs, src_mask)

    lang_model_decoder_output, lang_model_generator_output = self.l_model.decode(tgt_sent)

    decoder_output.requires_grad_()
    lang_model_decoder_output.requires_grad_()
    deep_fusion_output = self.deep_fusion(decoder_output, lang_model_decoder_output)

    return deep_fusion_output





class RnnDeepFusionModel(torch.nn.Module):
  def __init__(self, setting, src_lang, tgt_lang, t_model, num_pred_more=0):
    super(RnnDeepFusionModel, self).__init__()
    self.src_lang = src_lang
    self.tgt_lang = tgt_lang
    self.t_model = t_model
    self.deep_fusion = module.deep_fusion.RnnDeepFusionModule(tgt_lang.vocab_size,
                                                           self.t_model.decoder.model_dim,
                                                           setting["model_vars"]["model_dim"],
                                                           dropout_p=setting["train_vars"]["dropout_p"])
    self.num_pred_more = num_pred_more


  def translate(self, input, max_target_length, tgt_sent=None):
    src_mask = (input==torch.zeros(input.size(), dtype=torch.long).to(device))
    encoder_outputs = self.t_model.encode(input)


    start_token = self.tgt_lang.vocab2index["SEQUENCE_START"]
    end_token = self.tgt_lang.vocab2index["SEQUENCE_END"]

    batch_size = input.size(0)
    flag = [False for i in range(batch_size)]
    decoder_word_outputs = torch.zeros((batch_size, max_target_length), dtype=torch.long).to(device)
    decoder_word_outputs[:, 0] = start_token
    decoder_prob_outputs = torch.zeros((batch_size, max_target_length, self.tgt_lang.vocab_size), dtype=torch.float).to(device)

    layer_cache = {}
    for i in range(self.t_model.decoder.num_layers):
      layer_cache[i] = {"kv_mask": None, "key": None, "value": None}

    # decode words one by one
    for i in range(1, max_target_length):

      ### pseudo predict loop start ###
      if self.num_pred_more > 0:
        pseudo_flag = copy.deepcopy(flag)
        pseudo_word_outputs = torch.zeros((batch_size, self.num_pred_more+1), dtype=torch.long).to(device)
        pseudo_prob_outputs = torch.zeros((batch_size, self.num_pred_more+1, self.tgt_lang.vocab_size), dtype=torch.float).to(device)
        pseudo_decoder_outputs = torch.zeros((batch_size, self.num_pred_more+1, self.t_model.decoder.model_dim), dtype=torch.float).to(device)
        tmp_layer_cache = copy.deepcopy(layer_cache)

        for k in range(self.num_pred_more+1):
          # previous_word
          if k==0:
            if not torch.is_tensor(tgt_sent):
              decoder_input = decoder_word_outputs[:, i-1:i]
            else:
              decoder_input = tgt_sent[:, i-1:i]
          else:
            decoder_input = pseudo_word_outputs[:, k-1:k]
          decoder_output = self.t_model.decoder(decoder_input, encoder_outputs, src_mask, time_step=i-1+k, layer_cache=tmp_layer_cache)
          generator_output = self.t_model.generator(decoder_output[:, -1])

          likelihood, index = generator_output.data.topk(1)
          index = index.squeeze(1)

          for j, each in enumerate(index):
            if pseudo_flag[j]:
              index[j] = self.tgt_lang.vocab2index["PADDING"]
            elif int(each) == end_token:
              pseudo_flag[j] = True

          pseudo_word_outputs[:, k] = index
          pseudo_prob_outputs[:, k] = generator_output
          pseudo_decoder_outputs[:, k] = decoder_output[:, -1]

          if k==0:
            t_out = decoder_output[:, -1]
            layer_cache = copy.deepcopy(tmp_layer_cache)
        ### pseudo predict loop end ###

        #language model
        sort_index = torch.tensor(list(range(pseudo_decoder_outputs.size(1)-1, -1, -1)))
        inputs = pseudo_decoder_outputs[:, sort_index]

        deep_fusion_output = self.deep_fusion(t_out, inputs[:, :-1])

      else:
        raise

      likelihood, index = deep_fusion_output.data.topk(1)
      index = index.squeeze(1)

      for j, each in enumerate(index):
        if flag[j]:
          index[j] = self.tgt_lang.vocab2index["PADDING"]
        elif int(each) == end_token:
          flag[j] = True

      decoder_word_outputs[:, i] = index
      decoder_prob_outputs[:, i] = deep_fusion_output

      if all(flag):
        break

    return decoder_prob_outputs, decoder_word_outputs
