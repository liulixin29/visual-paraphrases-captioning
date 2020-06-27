import json
import cPickle
import numpy as np
from collections import defaultdict


def precook(s, n=4, out=False):
  """
  Takes a string as input and returns an object that can be given to
  either cook_refs or cook_test. This is optional: cook_refs and cook_test
  can take string arguments as well.
  :param s: string : sentence to be converted into ngrams
  :param n: int    : number of ngrams for which representation is calculated
  :return: term frequency vector for occuring ngrams
  """
  words = s.split()
  counts = defaultdict(int)
  for k in xrange(1,n+1):
    for i in xrange(len(words)-k+1):
      ngram = tuple(words[i:i+k])
      counts[ngram] += 1
  return counts


def modified_df(generated, df=None):
    num_sents = df['num_sents']
    df = df['document_frequency']
    all_scores = 0.0
    for i, caption in enumerate(generated.values()):
        ngrams = precook(caption, n=3)
        score = 0.0
        scores = dict()
        for ngram in ngrams:
            score += np.log((1.0 * num_sents) / df.get(ngram, 1.0)) * ngrams[ngram]
            scores[len(ngram)] = scores.get(len(ngram), 0.0) + np.log((1.0 * num_sents) / df.get(ngram, 1.0)) * ngrams[ngram]
        score = scores[1] / 35.166 + scores[2] / 65.01 + scores[2] / 79.76
        all_scores += score
    all_scores /= len(generated)
    return all_scores


class EvaluateDiversity(object):

    def __init__(self):
        self.all_bigram = dict()
        self.all_trigram = dict()
        self.all_unigram = dict()
        self.all_sentence = dict()
        self.bigram_num = 0
        self.trigram_num = 0
        self.unigram_num = 0
        self.sen_num = 0
        self.story_num = 0

    def diversity_evaluate(self, data, printvalue=True):
        sum_len = 0.0
        for string_ in data:
            sum_len += len(string_.strip().split())
            sen = string_

            splited = sen.split('.')

            for sp in splited:
                if sp.strip() != "":
                    self.all_sentence[sp] = 1
                    self.sen_num += 1
            self.story_num += 1
            sen_words = sen.strip().split()
            unigram = [sen_words[i] for i in range(len(sen_words))]
            if len(sen_words) >= 2:
                bigram = [sen_words[i] + sen_words[i + 1] for i in range(len(sen_words) - 2)]
            else:
                bigram = []
            if len(sen_words) >= 3:
                trigram = [sen_words[i] + sen_words[i + 1] + sen_words[i + 2] for i in
                           range(len(sen_words) - 3)]
            else:
                trigram = []
            for word in bigram:
                self.all_bigram[word] = 1
                self.bigram_num += 1
            for word in trigram:
                self.all_trigram[word] = 1
                self.trigram_num += 1
            for word in unigram:
                self.all_unigram[word] = 1
                self.unigram_num += 1
        if printvalue:
            print('number of captions', len(data))
            print('average length', sum_len/ len(data))

            if self.sen_num == 0:
                print("sentence number: " + str(self.sen_num) + " unique sentence number (Dist-S): " + str(
                    len(self.all_sentence)) + " unique sentence rate: " + str(0))
            else:
                print("sentence number: " + str(self.sen_num) + " unique sentence number (Dist-S): " + str(
                    len(self.all_sentence)) + " unique sentence rate: " + str(
                    len(self.all_sentence) / (1.0 * self.sen_num)))
            if self.unigram_num == 0:
                print("unigram number: " + str(self.unigram_num) + " unique unigram number (Dist-1): " + str(
                    len(self.all_unigram)) + " unique unigram rate: " + str(0))
            else:
                print("unigram number: " + str(self.unigram_num) + " unique unigram number (Dist-1): " + str(
                    len(self.all_unigram)) + " unique unigram rate: " + str(
                    len(self.all_unigram) / (1.0 * self.unigram_num)))
            if self.bigram_num == 0:
                print("bigram number: " + str(self.bigram_num) + " unique bigram number (Dist-2): " + str(
                    len(self.all_bigram)) + " unique bigram rate: " + str(0))
            else:
                print("bigram number: " + str(self.bigram_num) + " unique bigram number (Dist-2): " + str(
                    len(self.all_bigram)) + " unique bigram rate: " + str(len(self.all_bigram) / (1.0 * self.bigram_num)))
            if self.trigram_num == 0:
                print("trigram number: " + str(self.trigram_num) + " unique trigram number (Dist-3): " + str(
                    len(self.all_trigram)) + " unique trigram rate: " + str(0))
            else:
                print("trigram number: " + str(self.trigram_num) + " unique trigram number (Dist-3): " + str(
                    len(self.all_trigram)) + " unique trigram rate: " + str(
                    len(self.all_trigram) / (1.0 * self.trigram_num)))

        return len(self.all_unigram), len(self.all_bigram), len(self.all_trigram), len(self.all_sentence)


def evaluate(json_file):
    print(json_file)
    df = cPickle.load(open('df_processed.pkl', 'rb'))

    data = json.load(open(json_file))
    data = {item['image_id']:item['caption'] for item in data}

    ## evaluate conventional metrics
    # import sys
    # sys.path.append("coco-caption")
    # annFile = '../imgcap/annotations/captions_val2014.json'
    # from pycocotools.coco import COCO
    # from pycocoevalcap.eval import COCOEvalCap
    # coco = COCO(annFile)
    # cocoRes = coco.loadRes(json_file)
    # cocoEval = COCOEvalCap(coco, cocoRes)
    # cocoEval.params['image_id'] = cocoRes.getImgIds()
    # cocoEval.evaluate()

    gen_captions = list(data.values())
    e = EvaluateDiversity()
    e.diversity_evaluate(gen_captions)
    print('Tdiv:', modified_df(data, df))


if __name__ == '__main__':
    evaluate('Tdiv_0.1.json')