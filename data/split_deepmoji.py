import numpy as np
import os, sys

train_ratio = 0.8
dev_ratio = 0.1
test_ratio = 0.1

def read_data_file(input_file: str):
    vecs = np.load(input_file)
    np.random.seed(1)
    np.random.shuffle(vecs)
    tsize = vecs.shape[0]
    trsize = int(tsize * train_ratio)
    dsize = trsize + int(tsize * dev_ratio)
    print(vecs.shape, tsize, trsize, dsize, dsize-trsize, tsize-dsize)
    return vecs[:trsize], vecs[trsize:dsize], vecs[dsize:tsize]

if __name__ == '__main__':
    in_dir = sys.argv[1]
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    for split in ['pos_pos_sampled', 'pos_neg_sampled', 'neg_pos_sampled', 'neg_neg_sampled']:
        train, dev, test = read_data_file(in_dir + '/' + split + '.npy')
        for split_dir, data in zip(['train', 'dev', 'test'], [train, dev, test]):
            os.makedirs(out_dir + '/' + split_dir, exist_ok=True)
            np.save(out_dir + '/' + split_dir + '/' + split + '.npy', data)