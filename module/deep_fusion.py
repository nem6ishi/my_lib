import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class DeepFusionModule(torch.nn.Module):
  def __init__(self, vocab_size, t_model_dim, l_model_dim, model_dim, dropout_p=0.1):
    super(DeepFusionModule, self).__init__()

    self.controller = torch.nn.Linear(l_model_dim, l_model_dim)
    self.w = torch.nn.Linear(t_model_dim+l_model_dim, model_dim)
    self.dropout = torch.nn.Dropout(p=dropout_p)
    self.out = torch.nn.Linear(model_dim, vocab_size)

  def forward(self, t_input, l_input):
    l_input = self.controller(l_input) * l_input
    output = self.dropout(self.w(torch.cat((t_input, l_input), -1)))

    output = self.out(output)
    output = torch.nn.functional.log_softmax(output, dim=-1)

    return output




class RnnDeepFusionModule(torch.nn.Module):
  def __init__(self, vocab_size, t_model_dim, model_dim, dropout_p=0.1):
    super(RnnDeepFusionModule, self).__init__()

    self.rnn = torch.nn.GRU(t_model_dim, model_dim, num_layers=1, batch_first=True)
    self.controller = torch.nn.Linear(model_dim, model_dim)

    self.w = torch.nn.Linear(t_model_dim+model_dim, model_dim)
    self.out = torch.nn.Linear(model_dim, vocab_size)
    self.dropout = torch.nn.Dropout(p=dropout_p)

  def forward(self, t_input, n_more):

    rnn_output, rnn_hidden = self.rnn(n_more)
    rnn_hidden = rnn_hidden.squeeze(0).contiguous()

    l_input = self.controller(rnn_hidden) * rnn_hidden
    output = self.dropout(self.w(torch.cat((t_input, l_input), -1)))

    output = self.out(output)
    output = torch.nn.functional.log_softmax(output, dim=-1)

    return output
