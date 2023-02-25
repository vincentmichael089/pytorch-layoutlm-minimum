import torch
import torch.nn as nn
import pytorch_lightning as pl
import math

class LayoutLMEncoding(pl.LightningModule):
  def __init__(self, max_position, emb_size): 
    super(LayoutLMEncoding, self).__init__()
    self.mod_emb = int(emb_size/4)
    self.x_position_embeddings = nn.Embedding(max_position, self.mod_emb)
    self.y_position_embeddings = nn.Embedding(max_position, self.mod_emb)
  
  def forward(self, bbox):
    left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
    upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
    right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
    lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])

    return torch.cat([
      left_position_embeddings,
      upper_position_embeddings,
      right_position_embeddings,
      lower_position_embeddings
    ],
    dim=-1,)
  
class PositionalEncoding(pl.LightningModule):
  """Positional encoding."""
  def __init__(self, num_hiddens, dropout, max_len=1200):
    super(PositionalEncoding, self).__init__()
    self.dropout = nn.Dropout(dropout)
    self.register_buffer("P", torch.zeros((1, max_len, num_hiddens)))
    self.register_buffer("tempX", torch.arange(max_len, dtype=torch.float32, device = self.device).reshape(-1, 1) / 
      torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32, device = self.device) / num_hiddens))
    self.P[:, :, 0::2] = torch.sin(self.tempX)
    self.P[:, :, 1::2] = torch.cos(self.tempX)

  def forward(self, X):
    X = X + self.P[:, :X.shape[1], :]
    return self.dropout(X)
  
class TokenEmbedding(pl.LightningModule):
  def __init__(self, vocab_size: int, emb_size):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, emb_size)
    self.emb_size = emb_size

  def forward(self, tokens):
    return self.embedding(tokens.to(self.device)) * math.sqrt(self.emb_size)
  
class LayoutLMMinimumEncoderDecoder(pl.LightningModule):
  def __init__(self,
    num_encoder_layers: int,
    num_decoder_layers: int,
    emb_size: int,
    nhead: int,
    src_vocab_size: int,
    trg_vocab_size: int,
    dim_feedforward: int = 512,
    dropout: float = 0.1):
    super().__init__()
    self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
    self.tgt_tok_emb = TokenEmbedding(trg_vocab_size, emb_size)
    self.positional_encoding = PositionalEncoding(emb_size, dropout = dropout)
    self.positional_encoding_2d = LayoutLMEncoding(1100, emb_size) #somehow the normalisation is buggy so this was set to 1100 instead of 1000
    
    # transformers encoder
    encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=nn.functional.relu, layer_norm_eps= 1e-5, batch_first=True, norm_first=False)
    encoder_norm = nn.LayerNorm(emb_size, 1e-5)
    self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    # transformers decoder
    decoder_layer = nn.TransformerDecoderLayer(d_model=emb_size, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=nn.functional.relu, layer_norm_eps= 1e-5, batch_first=True, norm_first=False)
    decoder_norm = nn.LayerNorm(emb_size, 1e-5)
    self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

    self.linear = nn.Linear(emb_size, trg_vocab_size)

  def forward(self, src, char_padding_mask, trg, trg_mask, trg_padding_mask, cross_attn_padding_mask, bbox):
    trg_emb = self.positional_encoding(self.tgt_tok_emb(trg))
    src_emb = self.positional_encoding(self.src_tok_emb(src))
    posen_2d = self.positional_encoding_2d(bbox)
    src_emb = src_emb + posen_2d
    memory = self.encoder(src_emb, None, char_padding_mask)
    output = self.decoder(trg_emb, memory, trg_mask, None, trg_padding_mask, cross_attn_padding_mask)
    return self.linear(output)

  def encode(self, src, char_padding_mask):
    src_emb = self.positional_encoding(self.src_tok_emb(src))
    memory = self.encoder(src_emb, None, char_padding_mask)
    return memory

  def decode(self, trg, memory, trg_mask, trg_padding_mask, cross_attn_padding_mask):
    trg_emb = self.positional_encoding(self.tgt_tok_emb(trg))
    return self.decoder(trg_emb, memory, trg_mask, None, trg_padding_mask, cross_attn_padding_mask)
