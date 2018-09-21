import json
import os
import glob
import pyrouge

import metric as rouge
import operator
import logging
def sort_sents(file_path, save_to_path):
    filelist = glob.glob(file_path)
    for i, file in enumerate(filelist):
        #print(file)
        reader = open(file)
        fileJson = json.load(reader)
        article = fileJson['article']
        abstract = fileJson['abstract']

        # article_dir = os.path.join(temp_path, str(i))
        # if not os.path.exists(article_dir): os.mkdir(article_dir)
        #
        # sents_dir = os.path.join(article_dir, 'sents')
        # if not os.path.exists(sents_dir): os.mkdir(sents_dir)
        #
        # refs_dir = os.path.join(article_dir, 'abstracts')
        # if not os.path.exists(refs_dir): os.mkdir(refs_dir)

        rouge_final_f_scores_dict = {}

        for j, sent in enumerate(article):

            # sentence_file_dir = os.path.join(sents_dir, str(j))
            # if not os.path.exists(sentence_file_dir): os.mkdir(sentence_file_dir)
            #
            # reference_file_dir = os.path.join(refs_dir, str(j))
            # if not os.path.exists(reference_file_dir):os.mkdir(reference_file_dir)
            #
            # sentence_file = os.path.join(sentence_file_dir, str(j) + '_sentence.txt')
            # with open(sentence_file, 'w') as f:
            #     f.write(sent)
            # reference_file = os.path.join(reference_file_dir, str(j) + '_reference.txt')
            # with open(reference_file, 'w') as f:
            #     f.write(abstract[0])
            # r = pyrouge.Rouge155()
            # r.model_filename_pattern = '#ID#_reference.txt'
            # r.system_filename_pattern = '(\d+)_sentence.txt'
            # r.model_dir = reference_file_dir
            # r.system_dir = sentence_file_dir
            # logging.getLogger('global').setLevel(logging.WARNING)  # silence pyrouge logging
            # rouge_results = r.convert_and_evaluate()
            # result_dict = r.output_to_dict(rouge_results)
            #
            # rouge_1_f_score = result_dict['rouge_1_f_score']
            # rouge_2_f_score = result_dict['rouge_2_f_score']
            # rouge_l_f_score = result_dict['rouge_l_f_score']
            #
            # rouge_final_f_scores_dict[j] = rouge_1_f_score * alpha1 \
            #                              + rouge_2_f_score * alpha2 \
            #                            +rouge_l_f_score *alphal

            #print(rouge_l_f_score)
            sent_s = sent.split(' ')

            #print(sent_s)
            if len(abstract)!=1:
                print('this abstract have more than one sentence')
            abstract_s = abstract[0].split(' ')
            #print(abstract_s)
            words = []
            abs = []
            for w in sent_s:
                if w in ['', ',', '.', '`', "'", '-']:
                #if w in ['`', ',', '', '.', ':', '//']:
                    continue
                words.append(w)
            for w in abstract_s:
                if w in ['', ',', '.', '`', "'", '-']:
                #if w in ['`', "'", ',', '', '.']:
                    continue
                abs.append(w)
            rouge_l_f_score = rouge.compute_rouge_l(words, abs)
            #print(rouge_l_f_score)
            rouge_final_f_scores_dict[j] = round(rouge_l_f_score, 5)
        rouge_l_f_scores  = sorted(rouge_final_f_scores_dict.items(),key=operator.itemgetter(1),reverse=True)

        save_obj = dict()
        save_obj["article"] = article
        save_obj["abstract"] = abstract
        save_obj["sents_id"] = [rouge_l_f_scores[i][0] for i in range(len(rouge_final_f_scores_dict))]
        save_obj["scores"] = [rouge_l_f_scores[i][1] for i in range(len(rouge_final_f_scores_dict))]

        with open(os.path.join(save_to_path, "{0}.json".format(i)), 'w') as json_file:
            json.dump(save_obj, json_file, indent=4)

        if((i+1)%1000==0):
            print('processed {}file'.format(i+1))
train_file_path = '/media/yyl/e62cfea3-fff0-4e79-9e11-b0b00c401896/bine/data/train/*'
val_file_path = '/media/yyl/e62cfea3-fff0-4e79-9e11-b0b00c401896/bine/data/val/*'
train_save_to_path = '/media/yyl/e62cfea3-fff0-4e79-9e11-b0b00c401896/bine/data2/train'
val_save_to_path = '/media/yyl/e62cfea3-fff0-4e79-9e11-b0b00c401896/bine/data2/val'
#temp_path = '/media/yyl/e62cfea3-fff0-4e79-9e11-b0b00c401896/bine/temp'

sort_sents(val_file_path, val_save_to_path)
sort_sents(train_file_path, train_save_to_path)
