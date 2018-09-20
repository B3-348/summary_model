import json
import os

SAVE_DATA_PATH = "/media/yyl/e62cfea3-fff0-4e79-9e11-b0b00c401896/bine/data"


def write_test_file_to_json(file_path, json_file_path):
    """
    This method is to write out json file for the test file
    """
    with open(file_path) as f:
        count = 0
        for line in f.readlines():
            save_obj = dict()

            line = line.strip()

            # get the index of content, id
            index1 = line.index("`` content '' :")
            index1_end = index1 + len("`` content '' :")
            index2 = line.index("`` id '' :")
            index2_end = index2 + len("`` id '' :")
            index3 = line.rindex("-RCB-")

            content = line[index1_end + 1:index2 - 1].strip()
            nid = line[index2_end:index3].strip()

            content = content[2:-4].strip().lower()

            # do twice to get the raw id
            nid = nid.strip()[1:-1].strip()
            nid = nid.strip()[1:-1].strip()

            article = []

            # whether remove '\'
            content = content.replace("\\", "")

            contents = content.split(" . ")[:-1]
            for c in contents:
                if len(c.replace(" ", "").strip()) != 0:
                    article.append(c)
            if len(contents) == 0:
                article = [content]

            save_obj["article"] = article
            save_obj["id"] = nid

            with open(os.path.join(json_file_path, "valid_{}.json".format(count)), 'w') as json_file:
                json.dump(save_obj, json_file, indent=4)

            count += 1


def write_out_file(file_list, split):
    assert split in ['train', 'val', 'test']
    # transform train data_file
    count = 0
    for file in file_list:
        with open(file) as f:
            for line in f.readlines():
                save_obj = dict()

                line = line.strip()
                line = line[1:-1]
                # get the index of content, id, and title
                index1 = line.index("`` content '' :")
                index1_end = index1 + len("`` content '' :")
                index2 = line.index("`` id '' :")
                index2_end = index2 + len("`` id '' :")
                index3 = line.index("`` title '' :")
                index3_end = index3 + len("`` title '' :")

                content = line[index1_end + 1:index2 - 1].strip()
                id = line[index2_end:index3].strip()
                title = line[index3_end + 1:].strip()

                content = content[2:-4].strip().lower().replace(".", " . ").replace("!", " . ").replace("?", " . ")

                id = id[:-1].strip()
                title = title[2:-7].strip().lower()

                article = []

                contents = content.split(" . ")[:-1]
                # print(content)
                for c in contents:
                    if len(c.replace(" ", "").strip()) != 0 and c.strip().find(" ") != -1:
                        # print(c)
                        c = c + " . "
                        article.append(c)

                if len(contents) == 0:
                    contents = content.split(":")
                    for c in contents:
                        if len(c.replace(" ", "").strip()) != 0 and c.strip().find(" ") != -1:
                            # print(c)
                            c = c + " . "
                            article.append(c)

                save_obj["article"] = article
                save_obj["id"] = id
                save_obj["abstract"] = [title]

                with open(os.path.join(SAVE_DATA_PATH, "{0}/{1}.json".format(split, count)), 'w') as json_file:
                    json.dump(save_obj, json_file, indent=4)

                count += 1
                if count % 1000 == 0:
                    print("processing {0} files".format(count))


def convert_dataSet(train_file_list, val_file_list, test_file_list):

    assert len(train_file_list) > 0
    write_out_file(train_file_list, "train")

    assert len(val_file_list) > 0
    write_out_file(val_file_list, "val")

    if len(test_file_list) > 0:
        write_out_file(test_file_list, "test")
train_file_list = ['/media/yyl/e62cfea3-fff0-4e79-9e11-b0b00c401896/bine/tokenizer/bytecup.corpus.train.{}.txt'.format(i) for i in range(8)]
val_file_list = ['/media/yyl/e62cfea3-fff0-4e79-9e11-b0b00c401896/bine/tokenizer/bytecup.corpus.train.8.txt']
test_file = '/media/yyl/e62cfea3-fff0-4e79-9e11-b0b00c401896/bine/tokenizer/bytecup.corpus.validation_set.txt'
#convert_dataSet(train_file_list, val_file_list, test_file_list)

write_test_file_to_json(test_file, SAVE_DATA_PATH + '/test')
