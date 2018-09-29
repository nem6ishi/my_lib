import torch, random, time, copy, os, numpy
import language

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SingleCorpus:
  def __init__(self, lang, file_path, max_length=0): ### set max_length less than 0 to disable it
    if not os.path.isfile(file_path):
      raise ValueError("File does not exist: {0}".format(file_path))

    self.lang = lang
    self.file_path = file_path
    self.max_length = max_length

    self.corpus_size = 0
    self.sentences = {}
    self.lengths = {}
    self.is_not_converted = {}

    self.remove_indexes_set = self.check_length()


  def check_length(self):
    remove_indexes_set = set()
    if self.max_length > 0:
      with open(self.file_path, "r", encoding='utf-8') as file:
        for i, sent in enumerate(file):
          len_sent = sent.count(' ') + 1
          if len_sent > self.max_length:
            remove_indexes_set.add(i)
      self.corpus_size = i + 1
    else:
      self.corpus_size = sum(1 for line in open(self.file_path))
    return remove_indexes_set


  def import_file(self):
    self.corpus_size = 0
    with open(self.file_path, "r", encoding='utf-8') as file:
      for i, sent in enumerate(file):
        if i not in self.remove_indexes_set:
          self.sentences[self.corpus_size] = sent ### save sentence as str, and convert it later
          self.lengths[self.corpus_size] = sent.count(' ') + 1 + 2 ### for "SEQUENCE_START" and "SEQUENCE_END"
          self.is_not_converted[self.corpus_size] = True
          self.corpus_size += 1
        else:
          self.remove_indexes_set.remove(i)

  def convert_into_indexes(self, sentence_index_list):
    for idx in sentence_index_list:
      if self.is_not_converted[idx]:
        self.sentences[idx] = self.lang.sentence2indexes(self.sentences[idx].split())
        self.is_not_converted[idx] = False
    return 0



class ParallelCorpus:
  def __init__(self, src_lang, tgt_lang, src_file_path, tgt_file_path, max_length=0):
    self.corpus_size = 0
    self.src_corpus = SingleCorpus(src_lang, src_file_path, max_length)
    self.tgt_corpus = SingleCorpus(tgt_lang, tgt_file_path, max_length)

    if self.src_corpus.corpus_size != self.tgt_corpus.corpus_size:
      raise ValueError("Inappropriate pallalel corpus.")
    combined_remove_indexes_set = self.src_corpus.remove_indexes_set | self.tgt_corpus.remove_indexes_set
    self.src_corpus.remove_indexes_set = combined_remove_indexes_set
    self.tgt_corpus.remove_indexes_set = combined_remove_indexes_set


  def import_file(self):
    self.src_corpus.import_file()
    self.tgt_corpus.import_file()
    self.corpus_size = self.src_corpus.corpus_size


  def convert_into_indexes(self, sentence_index_list):
    self.src_corpus.convert_into_indexes(sentence_index_list)
    self.tgt_corpus.convert_into_indexes(sentence_index_list)
    return 0



class SingleBatch:
  def __init__(self):
    self.batch_size = 0
    self.sentences = []
    self.lengths = []
    self.masks = []


  def generate_random_indexes(self, corpus, batch_size):
    sample_idxs = numpy.random.randint(0, corpus.corpus_size, batch_size)
    return sample_idxs

  def generate_sequential_indexes(self, corpus, batch_size, num_iter):
    num_done = batch_size*num_iter
    num_rest = corpus.corpus_size - num_done
    if num_rest < self.batch_size:
      batch_size = num_rest
    sample_idxs = list(range(num_done, num_done+batch_size))
    return sample_idxs

  def add_padding_and_prepare_mask(self, corpus):
    max_length = max(self.lengths)
    for sent, length in zip(self.sentences, self.lengths):
      pad_len = max_length - length
      pad = [corpus.lang.vocab2index["PADDING"] for i in range(pad_len)]
      sent += pad
      mask = [0 for i in range(length)] + [1 for i in range(pad_len)]
      self.masks.append(mask)
    return 0


  def generate_batch(self, corpus, sample_idxs):
    self.__init__()
    self.batch_size = len(sample_idxs)
    corpus.convert_into_indexes(sample_idxs)

    batch_list= []
    for i, idx in enumerate(sample_idxs):
      sent = copy.deepcopy(corpus.sentences[idx])
      length = corpus.lengths[idx]
      self.sentences.append(sent)
      self.lengths.append(length)

    self.add_padding_and_prepare_mask(corpus)

    self.sentences = torch.LongTensor(self.sentences).to(device)
    self.masks = torch.ByteTensor(self.masks).to(device)

    return 0




class ParallelBatch:
  def __init__(self):
    self.src_batch = SingleBatch()
    self.tgt_batch = SingleBatch()

  def generate_random_indexes(self, corpus, batch_size):
    return self.src_batch.generate_random_indexes(corpus, batch_size)

  def generate_sequential_indexes(self, corpus, batch_size, num_iter):
    return self.src_batch.generate_sequential_indexes(corpus, batch_size, num_iter)

  def generate_batch(self, corpus, sample_idxs):
    self.src_batch.generate_batch(corpus.src_corpus, sample_idxs)
    self.tgt_batch.generate_batch(corpus.tgt_corpus, sample_idxs)
    return 0
