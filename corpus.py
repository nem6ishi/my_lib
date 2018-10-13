import torch, random, copy, os, numpy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


  def check_length(self): # check length before importing FOR parallel corpus
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
    self.corpus_size = 0 # reset corpus_size
    tmp_max_length = 0
    with open(self.file_path, "r", encoding='utf-8') as file:
      for i, sent in enumerate(file):
        if i not in self.remove_indexes_set:
          self.sentences[self.corpus_size] = sent ### save sentence as str, and convert it later
          self.lengths[self.corpus_size] = sent.count(' ') + 1 + 2 ### '2' for "SEQUENCE_START" and "SEQUENCE_END"
          tmp_max_length = max([tmp_max_length, self.lengths[self.corpus_size]])
          self.is_not_converted[self.corpus_size] = True
          self.corpus_size += 1
        else:
          self.remove_indexes_set.remove(i)
    self.max_length = tmp_max_length # update max_length to actual max_length

    return 0


  def convert_into_indexes(self, sentence_index_list):
    for idx in sentence_index_list:
      if self.is_not_converted[idx]:
        self.sentences[idx] = self.lang.sentence2indexes(self.sentences[idx].split())
        self.is_not_converted[idx] = False
    return 0



class ParallelCorpus:
  def __init__(self, src_lang, tgt_lang, src_file_path, tgt_file_path, max_length=0):
    self.corpus_size = 0
    self.share_corpus = True if (src_lang.path == tgt_lang.path and src_file_path == tgt_file_path) else False
    self.max_length = 0

    self.src_corpus = SingleCorpus(src_lang, src_file_path, max_length)
    if self.share_corpus:
      self.tgt_corpus = self.src_corpus
    else:
      self.tgt_corpus = SingleCorpus(tgt_lang, tgt_file_path, max_length)

      if self.src_corpus.corpus_size != self.tgt_corpus.corpus_size:
        raise ValueError("Inappropriate pallalel corpus.")
      combined_remove_indexes_set = self.src_corpus.remove_indexes_set | self.tgt_corpus.remove_indexes_set
      self.src_corpus.remove_indexes_set = copy.deepcopy(combined_remove_indexes_set)
      self.tgt_corpus.remove_indexes_set = copy.deepcopy(combined_remove_indexes_set)


  def import_file(self):
    self.src_corpus.import_file()
    if not self.share_corpus:
      self.tgt_corpus.import_file()
    if not self.src_corpus.corpus_size==self.tgt_corpus.corpus_size:
      raise
    self.corpus_size = self.src_corpus.corpus_size
    self.max_length = max([self.src_corpus.max_length, self.tgt_corpus.max_length])

    return 0


  def convert_into_indexes(self, sentence_index_list):
    self.src_corpus.convert_into_indexes(sentence_index_list)
    if not self.share_corpus:
      self.tgt_corpus.convert_into_indexes(sentence_index_list)
    return 0



class SingleBatch:
  def __init__(self, corpus, fixed_batch_size):
    self.fixed_batch_size = fixed_batch_size
    self.batch_size = 0
    self.corpus = corpus
    self.sample_idxs = []
    self.sentences = []
    self.lengths = []
    self.masks = []


  def generate_random_indexes(self):
    self.sample_idxs = numpy.random.randint(0, self.corpus.corpus_size, self.fixed_batch_size)
    return 0

  def generate_sequential_indexes(self, num_iter):
    num_done = self.fixed_batch_size * num_iter
    num_rest = self.corpus.corpus_size - num_done
    if num_rest < self.fixed_batch_size:
      batch_size = num_rest
    else:
      batch_size = self.fixed_batch_size
    self.sample_idxs = list(range(num_done, num_done+batch_size))
    return 0

  def add_padding_and_prepare_mask(self):
    max_length = max(self.lengths)
    for sent, length in zip(self.sentences, self.lengths):
      pad_len = max_length - length
      pad = [self.corpus.lang.vocab2index["PADDING"] for i in range(pad_len)]
      sent += pad
      mask = [0 for i in range(length)] + [1 for i in range(pad_len)]
      self.masks.append(mask)
    return 0


  def generate_batch(self):
    self.sentences = []
    self.lengths = []
    self.masks = []
    self.batch_size = len(self.sample_idxs)
    self.corpus.convert_into_indexes(self.sample_idxs)

    for i, idx in enumerate(self.sample_idxs):
      self.sentences.append(copy.deepcopy(self.corpus.sentences[idx]))
      self.lengths.append(self.corpus.lengths[idx])
    self.add_padding_and_prepare_mask()

    self.sentences = torch.LongTensor(self.sentences).to(device)
    self.masks = torch.ByteTensor(self.masks).to(device)

    return 0



class ParallelBatch:
  def __init__(self, parallel_corpus, fixed_batch_size):
    self.src_batch = SingleBatch(parallel_corpus.src_corpus, fixed_batch_size)
    self.tgt_batch = SingleBatch(parallel_corpus.tgt_corpus, fixed_batch_size)

  def generate_random_indexes(self):
    self.src_batch.generate_random_indexes()
    self.tgt_batch.sample_idxs = self.src_batch.sample_idxs
    return 0

  def generate_sequential_indexes(self, num_iter):
    self.src_batch.generate_sequential_indexes(num_iter)
    self.tgt_batch.sample_idxs = self.src_batch.sample_idxs
    return 0

  def generate_batch(self):
    self.src_batch.generate_batch()
    self.tgt_batch.generate_batch()
    return 0
