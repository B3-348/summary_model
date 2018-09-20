from data import Vocab
from models.ExtractorBatcher import Batcher
from collections import namedtuple


vocab = Vocab("/home/lemin/1TBdisk/PycharmProjects/summary-model/vocab_cnt.pkl", max_size=40000)  # create a vocabulary
# Create a batcher object that will create minibatches of data
hparam_list = ['mode', 'batch_size', 'threshold']
hps_dict = dict()
hps_dict['mode'] = 'train'
hps_dict['batch_size'] = 16
hps_dict['threshold'] = 0.2
hps = namedtuple("HParams", hps_dict.keys())(**hps_dict)
batcher = Batcher(data_path="/home/lemin/1TBdisk/data/bne/tokenizer/val/*", vocab=vocab, hps=hps,single_pass=False)
batch = batcher.next_batch()
print(len(batch.input_batch))
print(len(batch.target_batch))