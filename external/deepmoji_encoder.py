
from __future__ import division, unicode_literals

import json
from torchmoji.sentence_tokenizer import SentenceTokenizer
from torchmoji.model_def import torchmoji_feature_encoding
from torchmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH

import numpy as np

sent_race_dir = '/lt/work/shiva/class_imbalance/datasets/deepmoji/TwitterAAE-full-v1/data/processed/sent_race/'
out_dir = '/lt/work/shiva/class_imbalance/datasets/deepmoji/TwitterAAE-full-v1/data/processed/enc_sent_race/'

total = 100000

def get_sentences(d):
    with open(d + 'vocab', 'r') as f:
        vocab = f.readlines()
        vocab = map(lambda s: s.strip(), vocab)
    def to_words(sen):
        s = []
        for w in sen:
            s.append(vocab[w])
        return s
    with open(d + 'pos_pos', 'r') as f:
        pos_pos = f.readlines()
        pos_pos = [map(int, sen.split(' ')) for sen in pos_pos]
        pos_pos = pos_pos[:total]
        pos_pos = map(to_words, pos_pos)
    with open(d + 'pos_neg', 'r') as f:
        pos_neg = f.readlines()
        pos_neg = [map(int, sen.split(' ')) for sen in pos_neg]
        pos_neg = pos_neg[:total]
        pos_neg = map(to_words, pos_neg)
    with open(d + 'neg_pos', 'r') as f:
        neg_pos = f.readlines()
        neg_pos = [map(int, sen.split(' ')) for sen in neg_pos]
        neg_pos = neg_pos[:total]
        neg_pos = map(to_words, neg_pos)
    with open(d + 'neg_neg', 'r') as f:
        neg_neg = f.readlines()
        neg_neg = [map(int, sen.split(' ')) for sen in neg_neg]
        neg_neg = neg_neg[:total]
        neg_neg = map(to_words, neg_neg)
    
    return pos_pos, pos_neg, neg_pos, neg_neg

maxlen = 150
batch_size = 32

print('Tokenizing using dictionary from {}'.format(VOCAB_PATH))
with open(VOCAB_PATH, 'r') as f:
    vocabulary = json.load(f)
st = SentenceTokenizer(vocabulary, maxlen)

print('Loading model from {}.'.format(PRETRAINED_PATH))
model = torchmoji_feature_encoding(PRETRAINED_PATH)
print(model)

def sent_join(sents):
    a = []
    for s in sents:
        try:
            a.append(' '.join([x.decode('utf-8') for x in s]))
        except:
            print s
    return a

def batch_encode(in_data, bs_size):
    encoded_data = []
    for i in range(0, len(in_data), bs_size):
        tokenized, _, _ = st.tokenize_sentences(in_data[i: i + bs_size])
        encoded_batch = model(tokenized)
        encoded_data.extend(encoded_batch)
    return np.array(encoded_data)

pos_pos, pos_neg, neg_pos, neg_neg = get_sentences(sent_race_dir)

pos_pos = sent_join(pos_pos)
pos_neg = sent_join(pos_neg)
neg_pos = sent_join(neg_pos)
neg_neg = sent_join(neg_neg)

print(pos_pos[:2])

for d, name in zip([pos_pos, pos_neg, neg_neg, neg_pos], ['pos_pos', 'pos_neg', 'neg_neg', 'neg_pos']):
    encoded_data = batch_encode(d, bs_size=1000)
    print(encoded_data.shape, name)
    np.save(out_dir + '/{}.npy'.format(name), encoded_data)
    print(encoded_data[0], encoded_data[0].shape)
  
def save_file(data, file_name):
    import io
    encodable_data = []
    with io.open(file_name, "w", encoding="utf-8") as my_file:
        for line in data:
            try:
                my_file.write(line.encode('utf-8') + '\n')
                encodable_data.append(line)
            except:
                pass
    return encodable_data

new_pos_pos = save_file(pos_pos, out_dir + '/pos_pos.txt')
new_pos_neg = save_file(pos_neg, out_dir + '/pos_neg.txt')
new_neg_pos = save_file(neg_pos, out_dir + '/neg_pos.txt')
new_neg_neg = save_file(neg_neg, out_dir + '/neg_neg.txt')


for d, name in zip([new_pos_pos, new_pos_neg, new_neg_neg, new_neg_pos],
                   ['pos_pos', 'pos_neg', 'neg_neg', 'neg_pos']):
    encoded_data = batch_encode(d, bs_size=1000)
    np.save(out_dir + '/{}.npy'.format(name), encoded_data)