# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This file contains some utility functions"""

import tensorflow as tf
import time
import os
import gensim

from os.path import basename

FLAGS = tf.app.flags.FLAGS


def get_config():
    """Returns config for tf.session"""
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    return config


def load_ckpt(saver, sess, ckpt_dir="train"):
    """
    Load checkpoint from the ckpt_dir (if unspecified,
    this is train dir) and restore it to saver and sess, waiting 10 secs in the case of failure. Also returns checkpoint name.
    """
    while True:
        try:
            latest_filename = "checkpoint_best" if ckpt_dir == "eval" else None
            ckpt_dir = os.path.join(FLAGS.log_root, ckpt_dir)
            ckpt_state = tf.train.get_checkpoint_state(ckpt_dir, latest_filename=latest_filename)
            tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
            saver.restore(sess, ckpt_state.model_checkpoint_path)
            return ckpt_state.model_checkpoint_path
        except:
            tf.logging.info("Failed to load checkpoint from %s. Sleeping for %i secs...", ckpt_dir, 10)
            time.sleep(10)


def get_word_vector_and_write_out(w2v_file, out_put_text_file):
    # word2vec.{dim}d.{vsize}k.bin
    attrs = basename(w2v_file).split('.')
    w2v = gensim.models.Word2Vec.load(w2v_file).wv
    with open(out_put_text_file, 'w') as out_file:
        for key in w2v.vocab:
            print(*w2v[key])
            break



if __name__ == '__main__':
    w2v_path = "/home/lemin/1TBdisk/PycharmProjects/fast_abs_rl/word_vector_bcc/word2vec.128d.866k.bin"
    get_word_vector_and_write_out(w2v_path, 'embedding.txt')
