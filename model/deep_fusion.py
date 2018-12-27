import sys, torch, copy, random

sys.path.append("/home/neishi/workspace/my_lib")
import module.deep_fusion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class DeepFusionModel(torch.nn.Module):
  def __init__(self, setting, tgt_lang, t_model, l_model):
    super(DeepFusionModel, self).__init__()
    self.t_model = t_model
    self.l_model = l_model
    self.deep_fusion = module.DeepFusionModule(tgt_lang.vocab_size,
                                               self.t_model.decoder.model_dim,
                                               self.l_model.decoder.model_dim,
                                               setting["model_vars"]["model_dim"])



  def translate(self, input, ):
    encoder_outputs = t_model.encode(input)
    
