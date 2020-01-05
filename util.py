# encoding=utf-8

import pandas as pd
import csv
import tensorflow as tf
from sklearn.metrics import classification_report

def _read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            lines.append(line)
        return lines


if __name__ == '__main__':

    with open("corpus/new_wbt_data/validation_1.tsv", 'r', encoding='utf-8') as inp:
        # headers = next(inp)
        te = pd.read_csv(inp, sep="\t")
        te['text'] = [i.replace('\n', ' ') for i in te['text']]
        te['text'] = [i.replace('> ', '') for i in te['text']]
    # test_csv = _read_tsv("corpus/new_wbt_data/test_1.tsv")

    print(te)
    tee = pd.DataFrame()
    tee['label'] = te['label']
    tee['content'] = te['text']
    print(tee)
    tee.to_csv('TextClassification-master/data/dev_wbt.csv', encoding='utf-8', index=False)