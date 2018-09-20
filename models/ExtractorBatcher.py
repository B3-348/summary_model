import queue as Queue
from random import shuffle
from threading import Thread
import time
import numpy as np
import tensorflow as tf
import models.NewData as data


class Example(object):
    """Class representing a train/val/test example for text summarization."""

    def __init__(self, sents, target, scores, abstract, vocab, hps):

        """Initializes the Example, performing tokenization and truncation to produce the encoder, decoder and target sequences, which are stored in self.
        Args:
          article: source text; a string. each token is separated by a single space.
          abstract_sentences: list of strings, one per abstract sentence. In each sentence, each token is separated by a single space.
          vocab: Vocabulary object
          hps: hyperparameters
        """
        self.hps = hps

        # Get ids of special tokens
        self.pad_id = vocab.word2id(data.PAD_TOKEN)

        self.original_sents = sents
        self.abstract = abstract[0]

        self.sents_input = []

        # Process the sentences
        self.article_len = len(sents)
        self.maxlen_sents = max([len(sent.split()) for sent in sents])
        for sent in sents:
            sent_words = sent.split()
            sent_input = [vocab.word2id(w) for w in sent_words]
            while len(sent_input) < self.maxlen_sents:
                sent_input.append(self.pad_id)
            self.sents_input.append(sent_input)

        # Process the target
        self.target = [0] * self.article_len
        for ind, score in zip(target, scores):
            if (score > hps.threshold):
                self.target[ind] = 1
            else:
                break

    def pad_article(self, maxlen, pad_id):
        while self.article_len < maxlen:
            self.sents_input.append([pad_id] * self.maxlen_sents)

    def pad_target(self, maxlen, pad_id):
        while self.article_len < maxlen:
            self.target.append(pad_id)

class Batch(object):
    """Class representing a minibatch of train/val/test examples for text summarization."""

    def __init__(self, example_list, hps, vocab):
        """Turns the example_list into a Batch object.
        Args:
           example_list: List of Example objects
           hps: hyperparameters
           vocab: Vocabulary object
        """
        self.example_list = example_list
        self.pad_id = vocab.word2id(data.PAD_TOKEN)  # id of the PAD token used to pad sequences
        self.target_pad_id = 0
        self.init_encoder_seq(example_list, hps)  # initialize the input to the encoder
        self.init_decoder_seq(example_list, hps)  # initialize the input and targets for the decoder
        self.store_orig_strings(example_list)  # store the original strings

    def init_encoder_seq(self, example_list, hps):
        """Initializes the following:
            self.enc_batch:
              numpy array of shape (batch_size, <=max_enc_steps) containing integer ids (all OOVs represented by UNK id), padded to length of longest sequence in the batch
            self.enc_lens:
              numpy array of shape (batch_size) containing integers. The (truncated) length of each encoder input sequence (pre-padding).
            self.enc_padding_mask:
              numpy array of shape (batch_size, <=max_enc_steps), containing 1s and 0s. 1s correspond to real tokens in enc_batch and target_batch; 0s correspond to padding.
          If hps.pointer_gen, additionally initializes the following:
            self.max_art_oovs:
              maximum number of in-article OOVs in the batch
            self.art_oovs:
              list of list of in-article OOVs (strings), for each example in the batch
            self.enc_batch_extend_vocab:
              Same as self.enc_batch, but in-article OOVs are represented by their temporary article OOV number.
        """
        # Determine the maximum length of the encoder input sequence in this batch
        # max_enc_seq_len = max([ex.enc_len for ex in example_list])

        max_article_len = max([ex.article_len for ex in example_list])
        # Pad the encoder input sequences up to the length of the longest sequence
        for ex in example_list:
            # ex.pad_encoder_input(max_enc_seq_len, self.pad_id)
            ex.pad_article(max_article_len, self.pad_id)

        self.input_batch = []
        self.input_lens = []
        self.articles_padding_mask = np.zeros((hps.batch_size, max_article_len), dtype=np.float32)
        for i, ex in enumerate(example_list):
            self.input_batch.append(ex.sents_input)
            self.input_lens.append(ex.article_len)
            for j in range(ex.article_len):
                self.articles_padding_mask[i][j] = 1

    def init_decoder_seq(self, example_list, hps):
        """Initializes the following:
            self.dec_batch:
              numpy array of shape (batch_size, max_dec_steps), containing integer ids as input for the decoder, padded to max_dec_steps length.
            self.target_batch:
              numpy array of shape (batch_size, max_dec_steps), containing integer ids for the target sequence, padded to max_dec_steps length.
            self.dec_padding_mask:
              numpy array of shape (batch_size, max_dec_steps), containing 1s and 0s. 1s correspond to real tokens in dec_batch and target_batch; 0s correspond to padding.
            """
        max_article_len = max([ex.article_len for ex in example_list])
        # Pad the inputs and targets
        for ex in example_list:
            ex.pad_target(max_article_len, self.target_pad_id)

        self.target_batch = []
        for ex in example_list:
            self.target_batch.append(ex.target)

    def store_orig_strings(self, example_list):
        """Store the original article and abstract strings in the Batch object"""
        self.original_articles = [ex.original_sents for ex in example_list]  # list of lists
        self.original_abstracts = [ex.abstract for ex in example_list]


class Batcher(object):
    """A class to generate minibatches of data. Buckets examples together based on length of the encoder sequence."""

    BATCH_QUEUE_MAX = 100  # max number of batches the batch_queue can hold

    def __init__(self, data_path, vocab, hps, single_pass):
        """Initialize the batcher. Start threads that process the data into batches.
        Args:
          data_path: tf.Example filepattern.
          vocab: Vocabulary object
          hps: hyperparameters
          single_pass: If True, run through the dataset exactly once (useful for when you want to run evaluation on the dev or test set). Otherwise generate random batches indefinitely (useful for training).
        """
        self._data_path = data_path
        self._vocab = vocab
        self._hps = hps
        self._single_pass = single_pass

        # Initialize a queue of Batches waiting to be used, and a queue of Examples waiting to be batched
        self._batch_queue = Queue.Queue(self.BATCH_QUEUE_MAX)
        self._example_queue = Queue.Queue(self.BATCH_QUEUE_MAX * self._hps.batch_size)

        # Different settings depending on whether we're in single_pass mode or not
        if single_pass:
            self._num_example_q_threads = 1  # just one thread, so we read through the dataset just once
            self._num_batch_q_threads = 1  # just one thread to batch examples
            self._bucketing_cache_size = 1  # only load one batch's worth of examples before bucketing; this essentially means no bucketing
            self._finished_reading = False  # this will tell us when we're finished reading the dataset
        else:
            self._num_example_q_threads = 16  # num threads to fill example queue
            self._num_batch_q_threads = 4  # num threads to fill batch queue
            self._bucketing_cache_size = 100  # how many batches-worth of examples to load into cache before bucketing

        # Start the threads that load the queues
        self._example_q_threads = []
        for _ in range(self._num_example_q_threads):
            self._example_q_threads.append(Thread(target=self.fill_example_queue))
            self._example_q_threads[-1].daemon = True
            self._example_q_threads[-1].start()
        self._batch_q_threads = []
        for _ in range(self._num_batch_q_threads):
            self._batch_q_threads.append(Thread(target=self.fill_batch_queue))
            self._batch_q_threads[-1].daemon = True
            self._batch_q_threads[-1].start()

        # Start a thread that watches the other threads and restarts them if they're dead
        if not single_pass:  # We don't want a watcher in single_pass mode because the threads shouldn't run forever
            self._watch_thread = Thread(target=self.watch_threads)
            self._watch_thread.daemon = True
            self._watch_thread.start()

    def next_batch(self):
        """Return a Batch from the batch queue.
        If mode='decode' then each batch contains a single example repeated beam_size-many times; this is necessary for beam search.
        Returns:
          batch: a Batch object, or None if we're in single_pass mode and we've exhausted the dataset.
        """
        # If the batch queue is empty, print a warning
        if self._batch_queue.qsize() == 0:
            tf.logging.warning(
                'Bucket input queue is empty when calling next_batch. Bucket queue size: %i, Input queue size: %i',
                self._batch_queue.qsize(), self._example_queue.qsize())
            if self._single_pass and self._finished_reading:
                tf.logging.info("Finished reading dataset in single_pass mode.")
                return None

        batch = self._batch_queue.get()  # get the next Batch
        return batch

    def fill_example_queue(self):
        """Reads data from file and processes into Examples which are then placed into the example queue."""

        input_gen = self.text_generator(data.example_generator(self._data_path, self._single_pass))

        while True:
            try:
                (sents, abstract, sents_id, scores) = next(
                    input_gen)  # read the next example from file. article and abstract are both strings.
            except StopIteration:  # if there are no more examples:
                tf.logging.info("The example generator for this example queue filling thread has exhausted data.")
                if self._single_pass:
                    tf.logging.info(
                        "single_pass mode is on, so we've finished reading dataset. This thread is stopping.")
                    self._finished_reading = True
                    break
                else:
                    raise Exception("single_pass mode is off but the example generator is out of data; error.")

            # abstract_sentences = [sent.strip() for sent in data.abstract2sents(abstract)] # Use the <s> and </s> tags in abstract to get a list of sentences.
            example = Example(sents, sents_id, scores, abstract, self._vocab, self._hps)

            self._example_queue.put(example)  # place the Example in the example queue.

    def fill_batch_queue(self):
        """Takes Examples out of example queue, sorts them by encoder sequence length, processes into Batches and places them in the batch queue.
        In decode mode, makes batches that each contain a single example repeated.
        """
        while True:
            if self._hps.mode != 'decode':
                # Get bucketing_cache_size-many batches of Examples into a list, then sort
                inputs = []
                for _ in range(self._hps.batch_size * self._bucketing_cache_size):
                    inputs.append(self._example_queue.get())
                inputs = sorted(inputs, key=lambda inp: inp.article_len)  # sort by length of encoder sequence

                # Group the sorted Examples into batches, optionally shuffle the batches, and place in the batch queue.
                batches = []
                for i in range(0, len(inputs), self._hps.batch_size):
                    batches.append(inputs[i:i + self._hps.batch_size])
                if not self._single_pass:
                    shuffle(batches)
                for b in batches:  # each b is a list of Example objects
                    self._batch_queue.put(Batch(b, self._hps, self._vocab))

            else:  # beam search decode mode
                ex = self._example_queue.get()
                b = [ex for _ in range(self._hps.batch_size)]
                self._batch_queue.put(Batch(b, self._hps, self._vocab))

    def watch_threads(self):

        """Watch example queue and batch queue threads and restart if dead."""
        while True:
            time.sleep(60)
            for idx, t in enumerate(self._example_q_threads):
                if not t.is_alive():  # if the thread is dead
                    tf.logging.error('Found example queue thread dead. Restarting.')
                    new_t = Thread(target=self.fill_example_queue)
                    self._example_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()
            for idx, t in enumerate(self._batch_q_threads):
                if not t.is_alive():  # if the thread is dead
                    tf.logging.error('Found batch queue thread dead. Restarting.')
                    new_t = Thread(target=self.fill_batch_queue)
                    self._batch_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()

    ##############################################
    # rewrite text_generator function, get the article_text and abstract_text,change the article(some sentences) to an integral whole
    ##############################################
    def text_generator(self, example_generator):

        while True:
            sents, abstract, sents_id, scores = next(example_generator)
            # article_text, abstract_text = next(example_generator)

            if len(sents) == 0:
                # See https://github.com/abisee/pointer-generator/issues/1
                tf.logging.warning('Found an example with empty article text. Skipping it.')
            else:
                yield (sents, abstract, sents_id, scores)

