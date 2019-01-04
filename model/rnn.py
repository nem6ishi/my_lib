import sys, torch, copy, random

sys.path.append("/home/neishi/workspace/my_lib")
import module.rnn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class LanguageModel(torch.nn.Module):
  def __init__(self, setting, tgt_lang):
    super(LanguageModel, self).__init__()
    self.tgt_lang = tgt_lang
    self.lang_model =  module.rnn.FixedLengthRnnLanguageModel(tgt_lang.vocab_size,
                                                              setting["model_vars"]["emb_dim"],
                                                              setting["model_vars"]["model_dim"],
                                                              setting["model_vars"]["num_layers"],
                                                              dropout_p=setting["train_vars"]["dropout_p"],
                                                              padding_idx=tgt_lang.vocab2index["PADDING"])
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

      prob_output = self.lang_model(input_for_lang_model)
      prob_outputs[:, i] = prob_output

    prob_outputs = prob_outputs[:, reverse_idx] #backward

    return prob_outputs
