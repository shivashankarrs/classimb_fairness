import sys
import numpy as np
import os, math

black = 0.18
white = 0.82

pos = 110400 #0.69
neg = 49600 #0.31

pos_pos_size, pos_neg_size, neg_pos_size, neg_neg_size = [math.ceil(pos*black), math.ceil(pos*white), math.ceil(neg*black), \
                                                          math.ceil(neg*white)]

print(pos_pos_size, pos_neg_size, neg_pos_size, neg_neg_size)

def save_file(data, file_name):
    with open(file_name, 'w') as f:
        for item in data:
            f.write("%s" % item)

def downsample(input_file: str, raw_file, size):
    vecs = np.load(input_file)
    np.random.seed(1)
    np.random.shuffle(vecs)
    
    raw_data = get_sentences(raw_file)
    np.random.seed(1)
    np.random.shuffle(raw_data)
    return vecs[:size], raw_data[0:size]

def get_sentences(sentfile):
    with open(sentfile, 'r') as f:
        sentences = f.readlines()
    return sentences

if __name__ == '__main__':
    #arguments = docopt(__doc__)
    in_dir = sys.argv[1]
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    for split, size in zip(['pos_pos', 'pos_neg', 'neg_pos', 'neg_neg'], [pos_pos_size, pos_neg_size, neg_pos_size, neg_neg_size ]):
        data, raw = downsample(in_dir + '/' + split + '.npy', in_dir + '/' + split + '.txt', size)
        print(data.shape, len(raw), data[0])
        np.save(out_dir + '/' + split + '_sampled.npy', data)
        save_file(raw, out_dir + '/' + split + '_sampled.txt')