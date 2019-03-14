import gensim, torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def init_emb_layer(corpus, emb_dim, emb_layer):
  train_sents = []
  corpus_type = corpus.__class__.__name__

  if corpus_type == "ParallelCorpus":
    share_vocab = (corpus.src_lang==corpus.tgt_lang)

    if share_vocab:
      corpus.convert_into_indexes(list(range(corpus.corpus_size))) # this operation also adds special vocabs
      for i in range(corpus.corpus_size):
        train_sents.append([ str(word) for word in corpus.src_corpus.sentences[i] ])
        train_sents.append([ str(word) for word in corpus.tgt_corpus.sentences[i] ])
    else:
      raise

  else:
    raise


  w2v_model = gensim.models.word2vec.Word2Vec(train_sents, size=emb_dim, min_count=0, workers=10)

  for word_id in w2v_model.wv.vocab:
    emb_layer.weight.data[int(word_id)] = torch.from_numpy(w2v_model.wv[word_id]).to(device)

  return emb_layer
