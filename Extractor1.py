import tensorflow as tf
import json
import time
import numpy as np
import tensorflow.contrib as tc


def normalize(inputs,
              epsilon=1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs


def positional_encoding(inputs,
                        num_units,
                        zero_pad=True,
                        scale=True,
                        scope="positional_encoding",
                        reuse=None):
    '''Sinusoidal Positional_Encoding.

    Args:
      inputs: A 2d Tensor with shape of (N, T).
      num_units: Output dimensionality
      zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
      scale: Boolean. If True, the output will be multiplied by sqrt num_units(check details from paper)
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
        A 'Tensor' with one more rank than inputs's, with the dimensionality should be 'num_units'
    '''

    N, T = inputs.get_shape().as_list()
    with tf.variable_scope(scope, reuse=reuse):
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, 2.*i/num_units) for i in range(num_units)]
            for pos in range(T)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

        # Convert to a tensor
        lookup_table = tf.convert_to_tensor(position_enc)

        if zero_pad:
            lookup_table = tf.concat( (tf.zeros(shape=[1, num_units]),lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

        if scale:
            outputs = outputs * num_units**0.5

        return outputs


class Extractor(object):
    def __init__(self, batch, hps, vocab):
        self._batch = batch
        self._hps = hps
        self.vsize = vocab.size()
        self._vocab = vocab

        self.article = tf.placeholder(dtype=tf.float32, shape=[None, None, None], name="article")
        self.target = tf.placeholder(dtype=tf.float32, shape=[None, None], name="target")

        if self._hps.embedding:
            self.embedding_place = tf.placeholder(tf.float32, [self._vocab.size(), self._hps.emb_dim])
    def _embed_method(self, articles):
        """
        Args:
            article:[batch_size, max_sentences, max_words]
        """
        # batch_size list of [max_sentences, max_words] tensor
        with tf.variable_scope("embedding"):

            if self._hps.embedding is not None:
                embedding = tf.get_variable('embedding', dtype=tf.float32, initializer=self.embedding_place)
            else:
                embedding = tf.get_variable('embedding', [self.vsize, self._hps.emb_dim], dtype=tf.float32,
                                            initializer=self.trunc_norm_init)

            emb_inputs = [tf.nn.embedding_lookup(embedding, x) for x in tf.unstack(articles)]

            position_encode_inputs = [positional_encoding(x, self._hps.emb_dim) for x in tf.unstack(articles)]

            # batch_size list of 3-D tensor [max_sentences, max_words, embed_size]
            emb_inputs = [x + y for x, y in zip(emb_inputs, position_encode_inputs)]

            return emb_inputs


    def _conv_layer(self, input, output_channel, kernel_width, scope_name, dropout_rate=0.5):
        """
        Args:
             input: a 3D-tensor of (max_sentences, word_count, embed_dim)
             input_channels:
             output_channel: the output dimension of this cnn layer
             kernel_width: size of filter
             dropout_rate: the probability of dropout

        Returns:
            conv_output: a 3-D tensor of shape (max_sentences, word_count, output_channel)
        """
        with tf.variable_scope(scope_name):
            # output:(max_sentences, word_count, output_channel)
            conv_output = tf.layers.conv1d(input, output_channel, kernel_width, strides=1, padding="SAME")

            # relu layer
            conv_output = tf.nn.relu(conv_output)  # (max_sentences, word_count, output_channel)

            # dropout
            conv_output = tf.nn.dropout(conv_output, keep_prob=dropout_rate)

            return conv_output

    def _reduce_conv_output(self, conv_output):
        """
        This method is to reduce max_words dim use sum method

        Args:
            conv_output: a 3-D tensor of shape [max_sentences, max_words, hidden_size]
        Returns:
            reduce_conv_output: a 2-D tensor of shape [max_sentences, hidden_size]
        """
        assert len(conv_output.shape) == 3

        reduce_conv_output = tf.reduce_sum(conv_output, axis=1)

        return reduce_conv_output

    def _conv_sentence_features(self):
        """
        build network to pick useful sentence from  all sentence of article
        :return:
        """

        with tf.variable_scope("pick_sentence"):
            self.rand_unif_init = tf.random_uniform_initializer(-self._hps.rand_unif_init_mag, self._hps.rand_unif_init_mag,seed=123)
            self.trunc_norm_init = tf.truncated_normal_initializer(stddev=self._hps.trunc_norm_init_std)
            #sentence_score, sentence_class = util.get_score_article(self._batch.enc_batch, self._batch._dec_batch,self._hps.max_num_sentence)
            with tf.variable_scope("classified_embedding"):
                classified_embedding = tf.get_variable('classified_embedding', [self.vsize, self._hps.emb_dim],initializer=self.trunc_norm_init)
                emb_article = tf.nn.embedding_lookup(classified_embedding, self._batch.enc_input)
            cnn_kwidths_list = self._hps.cnn_kwiths_list #[6,7,8]
            cnn_hidden_dim = self._hps.cnn_hidden_dim #[128,128,128]
            self.batch_features = []
            i=1
            #for id_article in tf.unstack(emb_article):
            if i>0:
                #  id_article# shape:[num_sentence,max_enc_steps,emb_dim]
                id_article = emb_article
                pre_v = tf.get_variable('pre_v',shape=[id_article.get_shape()[-1],cnn_hidden_dim[0]])
                pre_b = tf.get_variable('pre_b',shape=[cnn_hidden_dim[0]],dtype=tf.float32,initializer=tf.zeros_initializer)
                next_input = id_article * pre_v + pre_b #initlizer ,transfer input article shape to input shape:[5,100,128]
                for i in range(self._hps.cnn_layers):
                     with tf.variable_scope('conv_layer_'+str(i)):
                          in_dim = id_article.get_shape()[-1]
                          v = tf.get_variable('v',shape=[cnn_kwidths_list[i],in_dim,cnn_hidden_dim[i]],dtype=tf.float32,initializer=tf.random_normal_initializer(mean=0,stddev=0.1),trainable=True)
                          w = tf.nn.l2_normalize(v,[0,1])
                          b = tf.get_variable('b',shape=[cnn_hidden_dim[i]],dtype=tf.float32,initializer=tf.zeros_initializer(),trainable=True)
                          next_input = tf.nn.conv1d(value=next_input,filters=w,stride=1,padding="SAME")
                fea_w = tf.get_variable('pre_w',shape=[cnn_hidden_dim[-1],1],dtype=tf.float32,initializer=tf.random_normal_initializer(mean=0,stddev=0.1),trainable=True)
                fea_b = tf.get_variable('fea_b',dtype=tf.float32,initializer=tf.zeros_initializer,trainable=True)
                article_features = tf.nn.bias_add(tf.matmul(next_input,fea_w),fea_b)
                self.batch_features.append(article_features)
                tf.layers.conv2d()
        return self.batch_features

    @staticmethod
    def multihead_attention(queries,
                            keys,
                            num_units=None,
                            num_heads=8,
                            dropout_rate=0,
                            is_training=True,
                            causality=False,
                            scope="multihead_attention",
                            reuse=None):
        '''Applies multihead attention.

        Args:
          queries: A 3d tensor with shape of [N, T_q, C_q].
          keys: A 3d tensor with shape of [N, T_k, C_k].
          num_units: A scalar. Attention size.
          dropout_rate: A floating point number.
          is_training: Boolean. Controller of mechanism for dropout.
          causality: Boolean. If true, units that reference the future are masked.
          num_heads: An int. Number of heads.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.

        Returns
          A 3d tensor with shape of (N, T_q, C)
        '''
        with tf.variable_scope(scope, reuse=reuse):
            # Set the fall back option for num_units
            if num_units is None:
                num_units = queries.get_shape().as_list[-1]

            # Linear projections
            # batch_size, max_step, num_units
            Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # (N, T_q, C)
            # batch_size, max_step, num_units
            K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
            # batch_size, max_step, num_units
            V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)

            # Split and concat
            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

            # Multiplication self-attention
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

            # Scale
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

            # Key Masking
            key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
            key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
            # if tf.equal(key_masks, 0) is true return paddings, else outputs
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

            # Causality = Future blinding
            if causality:
                diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
                tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense()  # (T_q, T_k)
                masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

                paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
                outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

            # Activation
            outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

            # Query Masking
            query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
            query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
            outputs *= query_masks  # broadcasting. (N, T_q, C)

            # Dropouts
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

            # Weighted sum
            outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

            # Restore shape
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

            # Residual connection
            outputs += queries

            # Normalize
            outputs = normalize(outputs)  # (N, T_q, C)

        return outputs

    @staticmethod
    def feedforward(inputs,
                    num_units=[2048, 512],
                    scope="multihead_attention",
                    reuse=None):
        '''Point-wise feed forward net.

        Args:
          inputs: A 3d tensor with shape of [N, T, C].
          num_units: A list of two integers.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.

        Returns:
          A 3d tensor with the same shape and dtype as inputs
        '''
        with tf.variable_scope(scope, reuse=reuse):
            # Inner layer
            params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                      "activation": tf.nn.relu, "use_bias": True}
            outputs = tf.layers.conv1d(**params)

            # Readout layer
            params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                      "activation": None, "use_bias": True}
            outputs = tf.layers.conv1d(**params)

            # Residual connection
            outputs += inputs

            # Normalize
            outputs = normalize(outputs)

        return outputs

    def _predicted(self, transform_output, scope_name="predicted"):
        """
        Args:
            transform_output: a 2-D tensor of shape [batch_size, attn_size]
        Output:
            predicted: a 2-D tensor of shape [batch_size, attn_size]
        """
        with tf.variable_scope(scope_name):
            assert len(transform_output.shape) == 2

            attn_size = transform_output.shape[1]

            w = tf.get_variable("w", [attn_size, attn_size], dtype=tf.float32, initializer=tc.layers.xavier_initializer())
            b = tf.get_variable("b", [attn_size], dtype=tf.float32)

            prediction = tf.matmul(transform_output, w) + b

            prediction = tf.nn.softmax(prediction)

            return prediction

    def _loss(self, preditcted, target):
        """
        Args:
            preditcted: a 2-D tensor of shape [batch_size, sentence_size]
            target: a 2-D tensor of shape (same as predicted)
        Returns:
            loss
        """
        # shape:[batch_size, sentence_size]
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=preditcted)

        sentence_lengths = tf.reduce_sum(target, axis=1)

        loss = loss/sentence_lengths

        average_loss = loss/self._hps.batch_size

        return average_loss

    def _train_op(self):
        article = self.article
        target = self.target

        assert len(target.shape) == 2

        output_units = target.shape[1]

        # batch_size list of embed_inputs [max_sentence, max_words, embed_size]
        embed_inputs = self._embed_method(article)

        # conv layer
        for filter_size in range(3, 6):
            embed_inputs = [self._conv_layer(x, self._hps.conv_output, filter_size) for x in embed_inputs]

        # batch_size list of 2-D tensor [max_sentences, conv_output]
        conv_outputs = [self._reduce_conv_output(x) for x in embed_inputs]

        # 3-D tensor [batch_size, max_sentences, conv_output]
        conv_outputs = tf.stack(conv_outputs, axis=0)

        attn_output = conv_outputs
        for i in range(self._hps.attn_layer):
            with tf.variable_scope("num_blocks_{}".format(i)):
                attn_output = self.multihead_attention(attn_output, attn_output, self._hps.attn_size)
                attn_output = self.feedforward(attn_output)  # shape [batch_size, max_sentence_length, attn_size]

        # [batch_size, max_sentences]
        attn_output = tf.reduce_mean(attn_output, axis=2)
        norm_attn_output = tf.nn.softmax(attn_output)
        self.score = norm_attn_output

        prediction = self._predicted(norm_attn_output)  # shape [batch_size, max_sentences]

        loss = self._loss(prediction, target)
        self.loss = loss

        tvars = tf.trainable_variables()
        gradients = tf.gradients(loss, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

        # Clip the gradients
        with tf.device("/gpu:0"):
            grads, global_norm = tf.clip_by_global_norm(gradients, self._hps.max_grad_norm)

        # Apply adagrad optimizer
        optimizer = tf.train.AdamOptimizer(self._hps.lr)
        with tf.device("/gpu:0"):
            self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step,
                                                       name='train_step')

    def build_graph(self):
        """Add the placeholders, models, global step, train_op and summaries to the graph"""
        tf.logging.info('Building graph...')
        t0 = time.time()
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        if self._hps.mode == 'train':
            self._train_op()

        t1 = time.time()
        tf.logging.info('Time to build graph: %i seconds', t1 - t0)

