import torch, copy, math
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class MultiHeadedAttention(torch.nn.Module):
  def __init__(self, num_head, model_dim):
    super(MultiHeadedAttention, self).__init__()

    assert model_dim % num_head == 0
    self.model_dim = model_dim
    self.num_head = num_head
    self.dim_per_head = model_dim // num_head

    self.linear_keys = torch.nn.Linear(model_dim, num_head * self.dim_per_head)
    self.linear_values = torch.nn.Linear(model_dim, num_head * self.dim_per_head)
    self.linear_querys = torch.nn.Linear(model_dim, num_head * self.dim_per_head)
    self.final_linear = torch.nn.Linear(model_dim, model_dim)


  def forward(self, query, key, value, q_mask, kv_mask, use_subseq_mask=False, layer_cache=None):
    batch_size = query.size(0)

    if layer_cache != None:
      if torch.is_tensor(layer_cache["kv_mask"]):
        kv_mask = torch.cat((layer_cache["kv_mask"], kv_mask), 1)
        key = torch.cat((layer_cache["key"], key), 1)
        value = torch.cat((layer_cache["value"], value), 1)

      layer_cache["kv_mask"] = kv_mask
      layer_cache["key"] = key
      layer_cache["value"] = value

    q_mask_here = q_mask.unsqueeze(1).unsqueeze(-1).expand(-1, self.num_head, -1, self.dim_per_head)
    query = self.linear_querys(query).view(batch_size, -1, self.num_head, self.dim_per_head).transpose(1, 2).masked_fill(q_mask_here, 0.0)

    kv_mask_here = kv_mask.unsqueeze(1).unsqueeze(-1).expand(-1, self.num_head, -1, self.dim_per_head)
    key = self.linear_keys(key).view(batch_size, -1, self.num_head, self.dim_per_head).transpose(1, 2).masked_fill(kv_mask_here, 0.0)
    value = self.linear_values(value).view(batch_size, -1, self.num_head, self.dim_per_head).transpose(1, 2).masked_fill(kv_mask_here, 0.0)

    x, attn_weights = self.attention(query, key, value, kv_mask, use_subseq_mask)
    x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_head * self.dim_per_head)
    x = self.final_linear(x)

    q_mask_here = q_mask.unsqueeze(-1).expand(-1, -1, x.size(-1))
    x = x.masked_fill(q_mask_here, 0.0)

    return x, attn_weights


  def attention(self, query, key, value, mask, use_subseq_mask=False):
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.dim_per_head)

    mask = mask.unsqueeze(1).unsqueeze(1).expand(-1, scores.size(1), scores.size(2), -1)
    if use_subseq_mask:
      mask = self.get_subseq_mask(mask.size())

    attn_weights = torch.nn.functional.softmax(scores.masked_fill(mask, float('-inf')), dim = -1)
    x = torch.matmul(attn_weights, value)
    return x, attn_weights


  def get_subseq_mask(self, size):
    k = 1 + size[-1] - size[-2]
    subsequent_mask = np.triu(np.ones(size), k=k).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask).to(device)
    return subsequent_mask



class PositionwiseFeedForward(torch.nn.Module):
  def __init__(self, model_dim, ff_dim, dropout_p=0.1):
    super(PositionwiseFeedForward, self).__init__()
    self.w_1 = torch.nn.Linear(model_dim, ff_dim)
    self.w_2 = torch.nn.Linear(ff_dim, model_dim)
    self.dropout = torch.nn.Dropout(dropout_p)

  def forward(self, x):
    return self.w_2(self.dropout(torch.nn.functional.relu(self.w_1(x))))



class TransformerEncoderLayer(torch.nn.Module):
  def __init__(self, model_dim, num_head, ff_dim, dropout_p=0.1):
    super(TransformerEncoderLayer, self).__init__()
    self.self_attn = MultiHeadedAttention(num_head, model_dim)
    self.layer_norm1 = torch.nn.LayerNorm(model_dim, eps=1e-6)
    self.feed_forward = PositionwiseFeedForward(model_dim, ff_dim, dropout_p)
    self.layer_norm2 = torch.nn.LayerNorm(model_dim, eps=1e-6)
    self.dropout = torch.nn.Dropout(dropout_p)

  def forward(self, input, mask):
    attn_out, _ = self.self_attn(input, input, input, mask, mask)
    out = self.layer_norm1(self.dropout(attn_out) + input)
    ff_out = self.feed_forward(out).masked_fill(mask.unsqueeze(-1).expand(-1, -1, out.size(-1)), 0.0)
    out = self.layer_norm2(self.dropout(ff_out) + out)
    return out



class RNNTransformerEncoder(torch.nn.Module):
  def __init__(self, vocab_size, emb_dim, model_dim, ff_dim, num_layers, num_head, bi_directional, dropout_p=0.1, padding_idx=0):
    super(RNNTransformerEncoder, self).__init__()

    self.vocab_size = vocab_size
    self.emb_dim = emb_dim
    self.model_dim = model_dim
    self.ff_dim = ff_dim
    self.num_layers = num_layers
    self.num_head = num_head
    self.bi_directional = bi_directional
    self.num_directions = 2 if bi_directional else 1

    self.dropout_p = dropout_p

    self.embedding = torch.nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=padding_idx)
    self.rnn = torch.nn.GRU(self.emb_dim, self.model_dim, num_layers=1, batch_first=True, bidirectional=self.bi_directional)

    if self.bi_directional:
      self.ff = torch.nn.Linear(self.model_dim*2, self.model_dim)

    self.layers = torch.nn.ModuleList([TransformerEncoderLayer(self.model_dim, self.num_head, self.ff_dim, self.dropout_p) for i in range(self.num_layers)])
    self.dropout = torch.nn.Dropout(self.dropout_p)


  def forward(self, input):
    mask = (input==torch.zeros(input.size(), dtype=torch.long).to(device))
    embedded = self.dropout(self.embedding(input)) # [B,T,H]

    x, _hidden = self.rnn(embedded)
    if self.bi_directional:
      x = self.ff(x)
    mask_here = mask.unsqueeze(-1).expand(-1, -1, x.size(-1))
    x = x.masked_fill(mask_here, 0.0)

    for layer in self.layers:
      x = layer(x, mask)

    return x



class TransformerDecoderLayer(torch.nn.Module):
  def __init__(self, model_dim, num_head, ff_dim, dropout_p=0.1):
    super(TransformerDecoderLayer, self).__init__()

    self.self_attn = MultiHeadedAttention(num_head, model_dim)
    self.layer_norm1 = torch.nn.LayerNorm(model_dim, eps=1e-6)
    self.context_attn = MultiHeadedAttention(num_head, model_dim)
    self.layer_norm2 = torch.nn.LayerNorm(model_dim, eps=1e-6)
    self.feed_forward = PositionwiseFeedForward(model_dim, ff_dim, dropout_p)
    self.layer_norm3 = torch.nn.LayerNorm(model_dim, eps=1e-6)
    self.dropout = torch.nn.Dropout(dropout_p)

  def forward(self, input, encoder_output, src_mask, tgt_mask, layer_cache=None):
    attn_out, _ = self.self_attn(input, input, input, tgt_mask, tgt_mask, use_subseq_mask=True, layer_cache=layer_cache)
    out = self.layer_norm1(self.dropout(attn_out) + input)

    context_out, _ = self.context_attn(out, encoder_output, encoder_output, tgt_mask, src_mask)
    out = self.layer_norm2(self.dropout(context_out) + out)

    ff_out = self.feed_forward(out).masked_fill(tgt_mask.unsqueeze(-1).expand(-1, -1, out.size(-1)), 0.0)
    out = self.layer_norm3(self.dropout(ff_out) + out)
    return out



class RNNTransformerDecoder(torch.nn.Module):
  def __init__(self, vocab_size, emb_dim, model_dim, ff_dim, num_layers, num_head, dropout_p=0.1, padding_idx=0):
    super(RNNTransformerDecoder, self).__init__()

    self.vocab_size = vocab_size
    self.emb_dim = emb_dim
    self.model_dim =  model_dim
    self.ff_dim = ff_dim
    self.num_layers = num_layers
    self.num_head = num_head

    self.dropout_p = dropout_p

    self.embedding = torch.nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=padding_idx)
    self.rnn = torch.nn.GRU(self.emb_dim, self.model_dim, num_layers=1, batch_first=True, bidirectional=False)
    self.layers = torch.nn.ModuleList([TransformerDecoderLayer(self.model_dim, self.num_head, self.ff_dim, self.dropout_p) for i in range(self.num_layers)])
    self.dropout = torch.nn.Dropout(self.dropout_p)


  def forward(self, input, encoder_output, src_mask, layer_cache=None, hidden=None):
    tgt_mask = (input == torch.zeros(input.size(), dtype=torch.long).to(device))
    embedded = self.dropout(self.embedding(input))

    if torch.is_tensor(hidden):
      x, hidden = self.rnn(embedded, hidden)
    else:
      x, hidden = self.rnn(embedded)

    mask_here = tgt_mask.unsqueeze(-1).expand(-1, -1, x.size(-1))
    x = x.masked_fill(mask_here, 0.0)

    for i, layer in enumerate(self.layers):
      x = layer(x, encoder_output, src_mask, tgt_mask,
                layer_cache=layer_cache[i] if layer_cache!=None else None)
    return x, hidden
