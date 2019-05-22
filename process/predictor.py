import torch, random, os, sys, re, glob, json, math, tensorboardX, datetime
from logging import getLogger
logger = getLogger(__name__)

sys.path.append("/home/neishi/workspace/my_lib")
import util.corpus, util.language, util.evaluation
from code import trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Predictor(trainer.Trainer):
  def __init__(self, setting):
    super().__init__(setting)
    self.step = self.setting["pred_vars"]["step"]


  def run(self):
    torch.set_num_threads(4)

    logger.info("Loading data")
    self.load_data()

    logger.info("Preparing model")
    self.prepare_model(mode="test")

    logger.info("Start prediction")
    self.prediction()


  def load_data(self):
    self.src_lang = util.language.Lang(self.setting["paths"]["src_vocab"])
    if self.setting["paths"]["src_vocab"] == self.setting["paths"]["tgt_vocab"]:
      self.tgt_lang = self.src_lang
    else:
      self.tgt_lang = util.language.Lang(self.setting["paths"]["tgt_vocab"])

    logger.info("Loading test corpus")
    self.test_corpus = util.corpus.ParallelCorpus(self.src_lang,
                                                  self.tgt_lang,
                                                  self.setting["paths"]["src_test"],
                                                  self.setting["paths"]["tgt_test"])
    self.test_corpus.import_file()


  def prediction(self):
    logger.info("Prediction")
    self.model.eval()
    prediction_sents = []
    batch_size = self.setting["pred_vars"]["batch_size"]
    pred_max_length = self.setting["pred_vars"]["prediction_max_length"]

    if self.setting["pred_vars"]["beam_size"] > 1 and batch_size != 1:
      logger.info("Set batch_size to 1 for beam search")
      batch_size = 1

    batch = util.corpus.ParallelBatch(self.test_corpus, batch_size)
    with torch.no_grad():
      for i in range(math.ceil(self.test_corpus.corpus_size / batch_size)):
        batch.generate_sequential_indexes(i)
        batch.generate_batch()

        outputs = self.model.encode(batch.src_batch.sentences)
        if self.setting["pred_vars"]["beam_size"] == 1:
          word_outputs, prob_outputs = self.model.decode(outputs,
                                                         batch.src_batch.masks,
                                                         pred_max_length,
                                                         reverse_output=self.setting["options"]["reverse_output"])
        else:
          word_outputs, prob_outputs = self.model.beam_search_decode(outputs,
                                                                     batch.src_batch.masks,
                                                                     pred_max_length,
                                                                     self.setting["pred_vars"]["beam_size"])

        for each in list(word_outputs):
          prediction_sents.append(self.tgt_lang.indexes2sentence(each))
        logger.info("{} sents done".format(batch_size*i + batch.batch_size))

    if self.setting["paths"]["tgt_test"]:
      logger.info("Save test prediction")
      save_dir = "./results"
      time_stamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
      pred_save_file = save_dir + "/test_prediction{}.txt".format(time_stamp)
      if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
      with open(pred_save_file, "w") as file:
        for sent in prediction_sents:
          sent = ' '.join(sent)
          file.write(sent + "\n")

      logger.info("Calculating bleu score")
      score, error = util.evaluation.calc_bleu(self.test_corpus.tgt_corpus.file_path,
                                               pred_save_file,
                                               self.setting["options"]["sentence_piece"],
                                               self.setting["options"]["use_kytea"],
                                               save_dir=save_dir,
                                               file_name="")
      logger.info("Bleu score: {} @step {}".format(score, self.step))
    else:
      for sent in prediction_sents:
        print(sent)

    return score, error
