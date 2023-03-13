

from io import open
import string
import random
import os
import sys

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, Vocab
from torch.nn.utils.rnn import pad_sequence
from collections import Counter

from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
import ecco
import spacy


# %cd machine_translation
sys.path.append('../src')
from dataloader import prepareData, unicodeToAscii, normalizeString, basic_prefixes

spacy_fr = spacy.load('fr_core_news_sm')
spacy_fr = spacy.load('en_core_web_sm')

MAX_SENTENCE_LENGTH = 20
FILTER_TO_BASIC_PREFIXES = False
SAVE_DIR = os.path.join(os.path.dirname('.'), 'models')

ENCODER_EMBEDDING_DIM = 256
ENCODER_HIDDEN_SIZE = 256
DECODER_EMBEDDING_DIM = 256
DECODER_HIDDEN_SIZE = 256

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('data/eng-fra.txt', 'r', encoding='utf-8') as f:
    lines = f.read().splitlines()

len(lines)
for example in random.choices(lines, k=5):
    pairs = example.split('\t')
    print(pairs[0], '<==>', pairs[1])

pairs[1], unicodeToAscii(pairs[1])
normalizeString(pairs[1])

pairs = prepareData(lines,
                    filter=True,
                    max_length=MAX_SENTENCE_LENGTH,
                    prefixes=basic_prefixes if FILTER_TO_BASIC_PREFIXES else ())

fr_tokenizer = get_tokenizer('spacy', language='fr')
en_tokenizer = get_tokenizer('spacy', language='en')

SPECIALS = ['<unk>', '<pad>', '<bos>', '<eos>']

en_list = []
fr_list = []
en_counter = Counter()
fr_counter = Counter()
en_lengths = []
fr_lengths = []
for en, fr in pairs:
    pass
    en = en_tokenizer(en)
    en_list.append(en)
    en_counter.update(en)
    en_lengths.append(len(en))
    fr = fr_tokenizer(fr)
    fr_list.append(fr)
    fr_counter.update(fr)
    fr_lengths.append(len(fr))

en_vocab = build_vocab_from_iterator(en_list, specials=SPECIALS)
fr_vocab = build_vocab_from_iterator(fr_list, specials=SPECIALS)

plt.hist(fr_lengths, rwidth=0.7, label='fr');
plt.hist(en_lengths, rwidth=0.5, label='en'); plt.legend(); plt.show()

plt.hist2d(fr_lengths, en_lengths, bins=MAX_SENTENCE_LENGTH-2, cmap='Blues'); plt.colorbar(); plt.show()

en_counter.most_common(5)
fr_counter.most_common(5)

VALI_PCT = 0.1
TEST_PCT = 0.1

train_data = []
valid_data = []
test_data = []

random.seed(6547)
for en, fr in pairs:
    pass
    en_tensor = torch.tensor([en_vocab[token] for token in en_tokenizer(en)])
    fr_tensor = torch.tensor([fr_vocab[token] for token in fr_tokenizer(fr)])
    random_draw = random.random()
    if random_draw <= VALI_PCT:
        valid_data.append((en_tensor, fr_tensor))
    elif random_draw <= VALI_PCT + TEST_PCT:
        test_data.append((en_tensor, fr_tensor))
    else:
        train_data.append((en_tensor, fr_tensor))

len(train_data), len(valid_data), len(test_data)
len(pairs)

PAD_IDX = en_vocab['<pad>']
BOS_IDX = en_vocab['<bos>']
EOS_IDX = en_vocab['<eos>']

for en_id, fr_id in zip(en_vocab.lookup_indices(SPECIALS), fr_vocab.lookup_indices(SPECIALS)):
  assert en_id == fr_id

def generate_batch(data_batch):
    en_batch, fr_batch = [], []
    for (en_item, fr_item) in data_batch:
        en_batch.append(torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))
        fr_batch.append(torch.cat([torch.tensor([BOS_IDX]), fr_item, torch.tensor([EOS_IDX])], dim=0))

    en_batch = pad_sequence(en_batch, padding_value=PAD_IDX, batch_first=False)
    fr_batch = pad_sequence(fr_batch, padding_value=PAD_IDX, batch_first=False)

    return en_batch, fr_batch

BATCH_SIZE = 16

train_iter = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch)
valid_iter = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=generate_batch)
test_iter = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=generate_batch)

for i, (eni, fri) in enumerate(train_iter):
    pass
    print(eni.shape, fri.shape)
    if i == 2:
        break
' '.join(en_vocab.lookup_tokens(eni[:, 0].tolist()))
' '.join(fr_vocab.lookup_tokens(fri[:, 0].tolist()))

## LSTM with Attention


from model import BahdanauEncoder, BahdanauDecoder, BahdanauAttentionQKV, BahdanauSeq2Seq, MultipleOptimizer, TransformerModel

enc = BahdanauEncoder(input_dim=len(en_vocab),
                      embedding_dim=ENCODER_EMBEDDING_DIM,
                      encoder_hidden_dim=ENCODER_HIDDEN_SIZE,
                      decoder_hidden_dim=DECODER_HIDDEN_SIZE,
                      dropout_p=0.15)

attn = BahdanauAttentionQKV(DECODER_HIDDEN_SIZE)

dec = BahdanauDecoder(output_dim=len(fr_vocab),
                      embedding_dim=DECODER_EMBEDDING_DIM,
                      encoder_hidden_dim=ENCODER_HIDDEN_SIZE,
                      decoder_hidden_dim=DECODER_HIDDEN_SIZE,
                      attention=attn,
                      dropout_p=0.15)

seq2seq = BahdanauSeq2Seq(enc, dec, device)

from utils import count_params, graph_forward
count_params(seq2seq)


from network import evaluate_transformer, train, evaluate, train_transformer
enc_optim = torch.optim.AdamW(seq2seq.encoder.parameters(), lr=1e-4)
dec_optim = torch.optim.AdamW(seq2seq.decoder.parameters(), lr=1e-4)
optims = MultipleOptimizer(enc_optim, dec_optim)
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)


N_EPOCHS = 2
CLIP = 10 # clipping value, or None to prevent gradient clipping
EARLY_STOPPING_EPOCHS = 2

if not os.path.exists(SAVE_DIR):
    print(f"Creating directory {SAVE_DIR}")
    os.mkdir(SAVE_DIR)

model_path = os.path.join(SAVE_DIR, 'bahdanau_en_fr.pt')
bahdanau_metrics = {}
best_valid_loss = float("inf")
early_stopping_count = 0
for epoch in tqdm(range(N_EPOCHS), leave=False, desc="Epoch"):
    pass
    train_loss = train(seq2seq, train_iter, optims, loss_fn, device, PAD_IDX, clip=CLIP)
    valid_loss = evaluate(seq2seq, valid_iter, loss_fn, device, PAD_IDX)

    if valid_loss < best_valid_loss:
        tqdm.write(f"Checkpointing at epoch {epoch + 1}")
        best_valid_loss = valid_loss
        torch.save(seq2seq.state_dict(), model_path)
        early_stopping_count = 0
    else:
        early_stopping_count += 1

    bahdanau_metrics[epoch+1] = dict(
        train_loss = train_loss,
        train_ppl = np.exp(train_loss),
        valid_loss = valid_loss,
        valid_ppl = np.exp(valid_loss)
    )

    if early_stopping_count == EARLY_STOPPING_EPOCHS:
        tqdm.write(f"Early stopping triggered in epoch {epoch + 1}")
        break

# seq2seq.load_state_dict(torch.load(model_path, map_location=device))
bahdanau_metrics_df = pd.DataFrame(bahdanau_metrics).T
plt.plot(bahdanau_metrics_df.train_loss, label='train')
plt.plot(bahdanau_metrics_df.valid_loss, label='valid'); plt.legend(); plt.show()
plt.plot(bahdanau_metrics_df.train_ppl, label='train')
plt.plot(bahdanau_metrics_df.valid_ppl, label='valid'); plt.legend(); plt.show()

best_valid_loss

from network import predict_text, show_attention


sentence = "i am a student"
sentence = "can we please go to the library ?"
sentence = 'her family moved away last year .'
result, attentions = predict_text(seq2seq, sentence, device, en_vocab, en_tokenizer, fr_vocab,  BOS_IDX, EOS_IDX)
print("Output >>>", result)
show_attention(sentence, result, attentions)

## Transformer encoder-decoder


transformer = TransformerModel(input_dim=len(en_vocab),
                             output_dim=len(fr_vocab),
                             d_model=256,
                             num_attention_heads=8,
                             num_encoder_layers=6,
                             num_decoder_layers=6,
                             dim_feedforward=2048,
                             max_seq_length=32,
                             pos_dropout=0.15,
                             transformer_dropout=0.3)

transformer = transformer.to(device)

count_params(transformer)

xf_optim = torch.optim.AdamW(transformer.parameters(), lr=1e-4)


N_EPOCHS = 2
CLIP = 15 # clipping value, or None to prevent gradient clipping
EARLY_STOPPING_EPOCHS = 5

model_path = os.path.join(SAVE_DIR, 'transformer_en_fr.pt')
transformer_metrics = {}
best_valid_loss = float("inf")
early_stopping_count = 0
for epoch in tqdm(range(N_EPOCHS), desc="Epoch"):
    train_loss = train_transformer(transformer, train_iter, xf_optim, loss_fn, device, PAD_IDX, clip=CLIP)
    valid_loss = evaluate_transformer(transformer, valid_iter, loss_fn, device, PAD_IDX)

    if valid_loss < best_valid_loss:
        tqdm.write(f"Checkpointing at epoch {epoch + 1}")
        best_valid_loss = valid_loss
        torch.save(transformer.state_dict(), model_path)
        early_stopping_count = 0
    elif epoch > EARLY_STOPPING_EPOCHS:
        early_stopping_count += 1

    transformer_metrics[epoch+1] = dict(
        train_loss = train_loss,
        train_ppl = np.exp(train_loss),
        valid_loss = valid_loss,
        valid_ppl = np.exp(valid_loss)
    )

    if early_stopping_count == EARLY_STOPPING_EPOCHS:
        tqdm.write(f"Early stopping triggered in epoch {epoch + 1}")
        break

best_valid_loss

def predict_transformer(text, model,
                        src_vocab=en_vocab,
                        src_tokenizer=en_tokenizer,
                        tgt_vocab=fr_vocab,
                        device=device):

    input_ids = [src_vocab[token] for token in src_tokenizer(text)]
    input_ids = [BOS_IDX] + input_ids + [EOS_IDX]

    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(input_ids).to(device).unsqueeze(1) # add fake batch dim

        causal_out = torch.ones(MAX_SENTENCE_LENGTH, 1).long().to(device) * BOS_IDX
        for t in range(1, MAX_SENTENCE_LENGTH):
            decoder_output = transformer(input_tensor, causal_out[:t, :])[-1, :, :]
            next_token = decoder_output.data.topk(1)[1].squeeze()
            causal_out[t, :] = next_token
            if next_token.item() == EOS_IDX:
                break

        pred_words = [tgt_vocab.lookup_token(tok.item()) for tok in causal_out.squeeze(1)[1:(t)]]
        return " ".join(pred_words)

predict_transformer("she is not my mother .", transformer)

# transformer = transformer.to('cpu')

# def _one_hot(token_ids, vocab_size, device=device):
#     return torch.zeros(token_ids.size(0), vocab_size).to(device).scatter_(1, token_ids, 1.)

# def get_embeds(embedding_matrix, position_embedding_layer, input_ids, device=device):
#     vocab_size = embedding_matrix.size(0)
#     one_hot_tensor = _one_hot(input_ids, vocab_size, device)

#     token_ids_tensor_one_hot = one_hot_tensor.to(device).clone().requires_grad_(True)
#     inputs_embeds = torch.matmul(token_ids_tensor_one_hot, embedding_matrix)
#     inputs_embeds = position_embedding_layer(inputs_embeds.unsqueeze(1))
#     return inputs_embeds, token_ids_tensor_one_hot

# def predict_and_visualize_transformer(input_text, model=transformer,
#                                       src_tokenizer=en_tokenizer,
#                                       tgt_tokenizer=fr_tokenizer,
#                                       src_vocab=en_vocab,
#                                       tgt_vocab=fr_vocab,
#                                       max_length=MAX_SENTENCE_LENGTH,
#                                       device=device):
#     input_text = normalizeString(input_text)
#     input_tokens = ['<bos>'] + src_tokenizer(input_text) + ['<eos>']
#     input_ids = [src_vocab[token] for token in input_tokens]
#     input_tensor = torch.tensor(input_ids).to(device).unsqueeze(1).to(device)

#     grads = []
#     i = 0
#     gen_tensor = torch.tensor([BOS_IDX]).unsqueeze(1).to(device)
#     while i < max_length:
#         # get embeddings and predict
#         src_key_padding_mask = (input_tensor == PAD_IDX).transpose(0, 1).to(device)
#         tgt_key_padding_mask = (gen_tensor == PAD_IDX).transpose(0, 1).to(device)

#         src_embed, token_tensor_one_hot = get_embeds(model.embed_src.weight, model.pos_enc, input_tensor, device)
#         tgt_embed, _ = get_embeds(model.embed_tgt.weight, model.pos_enc, gen_tensor, device)
#         logits = model(src_embeds=src_embed.to(device),
#                        tgt_embeds=tgt_embed.to(device),
#                        src_key_padding_mask=src_key_padding_mask,
#                        memory_key_padding_mask=src_key_padding_mask,
#                        tgt_mask = model.transformer.generate_square_subsequent_mask(gen_tensor.size(0)).to(device),
#                        tgt_key_padding_mask=tgt_key_padding_mask)

#         # extract next-word logits
#         next_logits = logits[-1].squeeze()
#         pred_id = next_logits.argmax()
#         pred_logit = next_logits.max()

#         # get gradient-based saliency for this token wrt input sequence
#         saliency = ecco.attribution.compute_saliency_scores(pred_logit,
#                                                             token_tensor_one_hot,
#                                                             src_embed)
#         grads.append(saliency['gradient'])

#         # update generated sequence with new token
#         gen_tensor = torch.cat([gen_tensor, pred_id.unsqueeze(0).unsqueeze(1)]).to(device)

#         i += 1

#         if pred_id == EOS_IDX:
#             break
#         if i == max_length:
#             break

#     gen_tokens = [tgt_vocab.lookup_token(i) for i in gen_tensor.squeeze()]
#     grad_array = np.stack(grads)

#     # plot it
#     fig = plt.figure(figsize=(8, 6))
#     ax = fig.add_subplot(111)
#     m = ax.matshow(grad_array, cmap='gray')
#     plt.xticks(range(grad_array.shape[1]), input_tokens, rotation=45, fontsize=12)
#     plt.yticks(range(grad_array.shape[0]), gen_tokens[1:], fontsize=12)
#     plt.colorbar(m)
#     plt.tight_layout()
#     plt.show()

# predict_and_visualize_transformer("I am going to the store.", device='cpu')
