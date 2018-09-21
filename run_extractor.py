import tensorflow as tf
import os

from models.Extractor import Extractor
from data import Vocab
from models.ExtractorBatcher import Batcher
from collections import namedtuple
from hyperparameters import FLAGS


def setup_training():


def main(unused_argv):
    if len(unused_argv) != 1:  # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)

    tf.logging.set_verbosity(tf.logging.INFO)  # choose what level of logging you want
    tf.logging.info('Starting seq2seq_attention in %s mode...', (FLAGS.mode))

    # Change log_root to FLAGS.log_root/FLAGS.exp_name and create the dir if necessary
    FLAGS.log_root = os.path.join(FLAGS.log_root, FLAGS.exp_name)
    if not os.path.exists(FLAGS.log_root):
        if FLAGS.mode == "train":
            os.makedirs(FLAGS.log_root)
        else:
            raise Exception("Logdir %s doesn't exist. Run in train mode to create it." % (FLAGS.log_root))

    vocab = Vocab(FLAGS.vocab_path, FLAGS.vocab_size)  # create a vocabulary

    # If in decode mode, set batch_size = beam_size
    # Reason: in decode mode, we decode one example at a time.
    # On each step, we have beam_size-many hypotheses in the beam, so we need to make a batch of these hypotheses.
    if FLAGS.mode == 'decode':
        FLAGS.batch_size = FLAGS.beam_size

    # If single_pass=True, check we're in decode mode
    if FLAGS.single_pass and FLAGS.mode != 'decode':
        raise Exception("The single_pass flag should only be True in decode mode")

    # Make a namedtuple hps, containing the values of the hyperparameters that the models needs
    hparam_list = ['mode', 'lr', 'adagrad_init_acc', 'rand_unif_init_mag', 'trunc_norm_init_std', 'max_grad_norm',
                   'hidden_dim', 'emb_dim', 'batch_size', 'max_dec_steps', 'max_enc_steps', 'coverage', 'cov_loss_wt',
                   'pointer_gen']
    hps_dict = {}
    for key, val in FLAGS.__flags.items():  # for each flag
        if key in hparam_list:  # if it's in the list
            hps_dict[key] = val  # add it to the dict
    hps = namedtuple("HParams", hps_dict.keys())(**hps_dict)

    # Create a batcher object that will create minibatches of data
    batcher = Batcher(FLAGS.data_path, vocab, hps, single_pass=FLAGS.single_pass)

    tf.set_random_seed(111)  # a seed value for randomness

    if hps.mode == 'train':
        print("creating models...")
        model = Extractor(hps, vocab)
        setup_training(model, batcher)
    elif hps.mode == 'eval':
        model = Extractor(hps, vocab)
        run_eval(model, batcher, vocab)
    elif hps.mode == 'decode':
        pass
    else:
        raise ValueError("The 'mode' flag must be one of train/eval/decode")


if __name__ == '__main__':
    tf.app.run()