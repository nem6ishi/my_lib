import os, copy

special_vocabs = ["PADDING", "UNK", "SEQUENCE_START", "SEQUENCE_END"]


class Lang:
  def __init__(self, self.path):
    self.path = self.path
    self.vocab2index = {}
    self.index2vocab = {}
    self.vocab_size = 0
    self.create_vocab()

  def add_word(self, word):
    if word not in self.vocab2index and len(word) > 0:
      self.vocab2index[word] = self.vocab_size
      self.index2vocab[self.vocab_size] = word
      self.vocab_size += 1

  def create_vocab(self):
    if not os.path.isfile(self.path):
      raise ValueError("File does not exist: {}".format(self.path))

    for word in special_vocabs:
      self.add_word(word)
    with open(self.path, "r", encoding='utf-8') as file:
      for line in file:
        word = line.split()[0]
        self.add_word(word)


  def sentence2indexes(self, sentence_as_list):
    v2i = self.vocab2index

    index_list = [v2i["SEQUENCE_START"]]
    for each in sentence_as_list:
      if each not in v2i:
        each = "UNK"
      index_list.append(v2i[each])
    index_list.append(v2i["SEQUENCE_END"])

    return index_list


  def indexes2sentence(self, index_list):
    sentence_as_list = []
    indexes = self.index2vocab.keys()

    for each in index_list:
      each = int(each)
      if each not in indexes:
        raise ValueError("Vocab index does not exist: {}".format(each))
      word = lang.index2vocab[each]
      if word != "PADDING":
        sentence_as_list.append(word)

    if sentence_as_list[0] == "SEQUENCE_START":
      sentence_as_list.pop(0)
    if len(sentence_as_list) > 0 and sentence_as_list[-1] == "SEQUENCE_END":
      sentence_as_list.pop()

    return sentence_as_list
