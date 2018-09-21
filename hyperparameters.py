import tensorflow as tf
FLAGS = tf.app.flags.FLAGS
# Where to find data
tf.app.flags.DEFINE_string('data_path', '/home/lemin/1TBdisk/data/bne/ntdata/train/*',
                           'Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')
tf.app.flags.DEFINE_string('vocab_path', '/home/lemin/1TBdisk/PycharmProjects/summary-model/vocab_cnt.pkl', 'Path expression to text vocabulary file.')

# Important settings
tf.app.flags.DEFINE_string('mode', 'train', 'must be one of train/eval/decode')
tf.app.flags.DEFINE_boolean('single_pass', False,
                            'For decode mode only. If True, run eval on the full dataset using a fixed checkpoint, i.e. take the current checkpoint, and use it to produce one summary for each example in the dataset, write the summaries to file and then get ROUGE scores for the whole dataset. If False (default), run concurrent decoding, i.e. repeatedly load latest checkpoint, use it to produce summaries for randomly-chosen examples and log the results to screen, indefinitely.')
tf.app.flags.DEFINE_float("lr_decay_rate", 0.95, 'learning rate decay number')
tf.app.flags.DEFINE_integer("impatient", 10, 'If val_loss or val_reward is bigger or smaller than the best_val impatient times, we will do a early stop')
# Where to save output
tf.app.flags.DEFINE_string('log_root', '/home/lemin/1TBdisk/PycharmProjects/summary-model/bne', 'Root directory for all logging.')
tf.app.flags.DEFINE_string('exp_name', 'test1',
                           'Name for experiment. Logs will be saved in a directory with this name, under log_root.')

# Extractor
tf.app.flags.DEFINE_integer('top_k', 2, 'the top_k sentences for extractor to extract')

# Hyperparameters
tf.app.flags.DEFINE_string('embedding', '', 'path to word2vector file')
tf.app.flags.DEFINE_integer('hidden_dim', 256, 'dimension of RNN hidden states')
tf.app.flags.DEFINE_integer('emb_dim', 128, 'dimension of word embeddings')
tf.app.flags.DEFINE_integer('batch_size', 16, 'minibatch size')
tf.app.flags.DEFINE_integer('max_enc_steps', 400, 'max timesteps of encoder (max source text tokens)')
tf.app.flags.DEFINE_integer('max_dec_steps', 100, 'max timesteps of decoder (max summary tokens)')
tf.app.flags.DEFINE_integer('beam_size', 4, 'beam size for beam search decoding.')
tf.app.flags.DEFINE_integer('min_dec_steps', 35,
                            'Minimum sequence length of generated summary. Applies only for beam search decoding mode')
tf.app.flags.DEFINE_integer('vocab_size', 40000,
                            'Size of vocabulary. These will be read from the vocabulary file in order. If the vocabulary file contains fewer words than this number, or if this number is set to 0, will take all words in the vocabulary file.')
tf.app.flags.DEFINE_float('lr', 0.15, 'learning rate')
tf.app.flags.DEFINE_float('adagrad_init_acc', 0.1, 'initial accumulator value for Adagrad')
tf.app.flags.DEFINE_float('rand_unif_init_mag', 0.02, 'magnitude for lstm cells random uniform inititalization')
tf.app.flags.DEFINE_float('trunc_norm_init_std', 1e-4, 'std of trunc norm init, used for initializing everything else')
tf.app.flags.DEFINE_float('max_grad_norm', 2.0, 'for gradient clipping')

# Pointer-generator or baseline models
tf.app.flags.DEFINE_boolean('pointer_gen', True, 'If True, use pointer-generator models. If False, use baseline models.')

# Coverage hyperparameters
tf.app.flags.DEFINE_boolean('coverage', False,
                            'Use coverage mechanism. Note, the experiments reported in the ACL paper train WITHOUT coverage until converged, and then train for a short phase WITH coverage afterwards. i.e. to reproduce the results in the ACL paper, turn this off for most of training then turn on for a short phase at the end.')
tf.app.flags.DEFINE_float('cov_loss_wt', 1.0,
                          'Weight of coverage loss (lambda in the paper). If zero, then no incentive to minimize coverage loss.')

# Utility flags, for restoring and changing checkpoints
tf.app.flags.DEFINE_boolean('convert_to_coverage_model', False,
                            'Convert a non-coverage models to a coverage models. Turn this on and run in train mode. Your current training models will be copied to a new version (same name with _cov_init appended) that will be ready to run with coverage flag turned on, for the coverage training stage.')
tf.app.flags.DEFINE_boolean('restore_best_model', False,
                            'Restore the best models in the eval/ dir and save it in the train/ dir, ready to be used for further training. Useful for early stopping, or if your training checkpoint has become corrupted with e.g. NaN values.')

# Debugging. See https://www.tensorflow.org/programmers_guide/debugger
tf.app.flags.DEFINE_boolean('debug', False, "Run in tensorflow's debug mode (watches for NaN/inf values)")
