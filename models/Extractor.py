
import  tensorflow as tf
import  numpy as np
import json
import util
from data import  Vocab
from batcher import Get_article
class config(object):
    rand_unif_init_mag = 0.02
    trunc_norm_init_std = 1e-4
    emb_dim = 100
    cnn_kwiths_list = [5,6,7]
    cnn_hidden_dim = [128,128,128]
    cnn_layers = 3





class Extractor(object):
    def __init__(self,batch,hps,vocab):
        self._batch = batch
        self._hps = hps
        self.vsize = vocab.size()


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
        return self.batch_features


    def attention_features(self):
        """
        get attention of each sentence in the article
        :return:
        """
        self.batch_attention = []
        with tf.get_variable("attention_features"):
            for article_features in self.batch_features:
                num_sentence = article_features.get_shape()[0]
                article_attention = tf.zeros(num_sentence,num_sentence)
                article_features =  tf.unstack(article_features,axis=0)
                for i,sentence_features in enumerate(article_features):
                     for j in range(i+1,num_sentence):
                         in_size = self.batch_features[0].get_shape()[-1]
                         v_att = tf.get_variable('v_att', shape=[in_size, in_size], dtype=tf.float32,initializer=tf.random_normal_initializer(mean=0, stddev=0.1), trainable=True)
                         pre_attention = tf.matmul(sentence_features,v_att)
                         sentence_attention = tf.matmul(pre_attention,article_features[j])
                         article_attention[i][j] = sentence_attention


    def predictions(self):
        pass

vocab_path = '/home/cy/Dataset/bytecup2018/vocab.json'
data_path ='/home/cy/Dataset/bytecup2018/train/*.json'

vocab = Vocab(vocab_path, 80)
get_article = Get_article(data_path=data_path,vocab=vocab)
config = config()

with tf.Session() as sess:
    example = get_article.next_example()
    extractor = Extractor(example,config,vocab)
    features = extractor._conv_sentence_features()
    sess.run(features)
    with open('/home/cy/features.json') as read_f:
        json.dump(features,read_f)


