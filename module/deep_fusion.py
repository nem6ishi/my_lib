import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class DeepFusionModule(torch.nn.Module):
  def __init__(self, vocab_size, t_model_dim, l_model_dim, model_dim):
    super(DeepFusionModule, self).__init__()

    self.controller = torch.nn.Linear(l_model_dim, l_model_dim)
    self.w = torch.nn.Linear(t_model_dim+l_model_dim, model_dim)
    self.dropout = torch.nn.Dropout(p=dropout_p)
    self.out = torch.nn.Linear(model_dim, vocab_size)

  def forward(self, t_input, l_input):
    l_input *= self.controller(l_input)
    output = self.w(torch.cat((t_input, l_input), -1))

    output = self.out(output.squeeze(1))
    output = torch.nn.functional.log_softmax(output, dim=1).squeeze(1)

    return output
