import os
import subprocess

data_dir = "/home/lemin/1TBdisk/data/bne"
file_list = ["bytecup.corpus.train.{}.txt".format(i) for i in range(0, 9)]
tokenizer_dir = "/home/lemin/1TBdisk/data/bne/tokenizer"


def tokenizer_stories(data_dir, tokenizer_dir):
    print("Preparing to tokenize {} to {}...".format(data_dir,
                                                     tokenizer_dir))

    # make IO list file
    print("Making list of files to tokenize...")
    with open("mapping.txt", "w") as f:
        for s in file_list:
            f.write(
                "{} \t {}\n".format(
                    os.path.join(data_dir, s),
                    os.path.join(tokenizer_dir, s)
                )
            )
    command = ['java', 'edu.stanford.nlp.process.PTBTokenizer',
               '-ioFileList', '-preserveLines', 'mapping.txt']
    print("Tokenizing {} files in {} and saving in {}...".format(
        len(file_list), data_dir, tokenizer_dir))
    subprocess.call(command)
    print("Stanford CoreNLP Tokenizer has finished.")
    os.remove("mapping.txt")

    num_orig = len(file_list)
    num_tokenized = len(os.listdir(tokenizer_dir))
    if num_orig != num_tokenized:
        raise Exception(
            "The tokenized stories directory {} contains {} files, but it "
            "should contain the same number as {} (which has {} files). Was"
            " there an error during tokenization?".format(
                tokenizer_dir, num_tokenized, data_dir, num_orig)
        )

    print("Successfully finished tokenizing {} to {}.\n".format(
        data_dir, tokenizer_dir))


def tokenize_single_file(file_name, tokenizer_dir):
    print("Preparing to tokenize {} to {}...".format(file_name,
                                                     tokenizer_dir))

    print("Making list of files to tokenize...")
    with open("mapping.txt", "w") as f:

        f.write("{} \t {}\n".format(
                os.path.join(data_dir, file_name),
                os.path.join(tokenizer_dir, file_name)
        ))
    command = ['java', 'edu.stanford.nlp.process.PTBTokenizer',
               '-ioFileList', '-preserveLines', 'mapping.txt']
    print("Tokenizing {} files in {} and saving in {}...".format(
        len(file_list), data_dir, tokenizer_dir))
    subprocess.call(command)
    print("Stanford CoreNLP Tokenizer has finished.")
    os.remove("mapping.txt")


if __name__ == '__main__':
    tokenizer_stories(data_dir, tokenizer_dir)