import sys
import torch
import numpy as np

try: sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
except: pass
# %cd transformers
sys.path.append('../src')

from utils import show_heatmaps, graph_forward

from model import Seq2SeqAttentionDecoder

vocab_size, embed_size, num_hiddens, num_layers, dropout = 10000, 32, 32, 2, 0.1
batch_size, num_steps = 4, 7

data = torch.MTFraEng(batch_size=128)

net = Seq2SeqAttentionDecoder(vocab_size, embed_size, num_hiddens, num_layers, dropout)

enc_outputs, hidden_state, enc_valid_lens = torch.zeros((batch_size, num_steps, num_hiddens)), torch.zeros((num_layers, batch_size, num_hiddens)), torch.ones(batch_size) * num_steps
state = [enc_outputs, hidden_state, enc_valid_lens]
X = torch.zeros((batch_size, num_steps), dtype=torch.long)

net(X, state)   # warm up






