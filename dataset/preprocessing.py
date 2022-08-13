import re
import os
import pandas as pd
import numpy as np
from tqdm import tqdm


def convert_to_task1():
    os.system('mkdir "./temp/train"')
    os.system('mkdir "./temp/dev"')
    os.system('mkdir "./temp/test"')

    os.system('python "./task1_converter_FIXED.py" "../deft_corpus-master/data/deft_files/train" "./temp/train"')
    os.system('python "./task1_converter_FIXED.py" "../deft_corpus-master/data/deft_files/dev" "./temp/dev"')
    os.system('xcopy "../deft_corpus-master/data/test_files/labeled/subtask_1" "./temp/test" /y /q')


def avramandrei_clean_data(sentence):
    """See https://github.com/avramandrei/UPB-SemEval-2020-Task-6/blob/77d92e9c386f270af6ed1db259d3ba6e8bde307b/task1/process.py#L9-L43"""

    sentence = re.sub("\[ ?link ?\][a-z]?( \( [a-z] \))?", "<link>", sentence)
    
    sentence = re.sub(r" ?https?:.+(\)|/|(\.pdf)|(\.PDF)|(\.html)|#| - U |aspx?|-[a-zA-z0-9]+|\.htm|\?.+)", "", sentence)
    sentence = re.sub(r"www.+?( |\))", "", sentence)
    sentence = sentence.replace(".  .", ".").replace(". .", ".").replace(", .", ".")

    sentence = sentence.replace("“ ", "\"").replace(" ”", "\"").replace("’", "'").replace("‘", "'").replace(",", ",").replace("⋅", "*")

    sentence = re.sub(r" size 12.+}", "", sentence)
    sentence = re.sub(r"5 \" MeV/\"c.+}", "", sentence)
    sentence = re.sub(r" } { }", "", sentence)

    sentence = re.sub(r"[^\s]+(\+|=|Δ|\*){1}[^\s]+", "<equation>", sentence)  # TODO: many equations don't get matched by the regex

    sentence = re.sub(r"^ (\d+ . )?", "", sentence)  # Remove the numerical index at the start of many sentences

    sentence = sentence.replace("do n't", "don't").replace("Do n't", "Don't")

    sentence = sentence.replace(" .", ".")
    sentence = sentence.replace(" ,", ",")
    sentence = sentence.replace(" ?", "?")
    sentence = sentence.replace(" - ", "-")
    sentence = sentence.replace("( ", "(")
    sentence = sentence.replace(" )", ")")
    sentence = sentence.replace(" & ", "&")
    sentence = sentence.replace(" ;", ";")
    sentence = sentence.replace(" '", "'")
    sentence = sentence.replace(" :", ":")
    sentence = sentence.replace(" $", "$")  # might be problematic: "I have $ 20" --> "I have$ 20"
    sentence = sentence.replace(" %", "%")
    sentence = re.sub(r"(_ )+", "", sentence)
    sentence = sentence.replace(",\"", "\"")

    return sentence


def load_dataset(directory):
    df = pd.DataFrame(columns=['SENTENCE', 'HAS_DEF', 'SOURCE', 'INDEX'])

    files = os.listdir(directory)
    for filename in tqdm(files):
        csv_df = pd.read_csv(os.path.join(directory, filename), sep="\t", header=None, encoding='utf-8',
                             names=["SENTENCE", "HAS_DEF"],
                             dtype={"SENTENCE": str, "HAS_DEF": np.uint8})
        csv_df['SOURCE'] = filename
        csv_df['INDEX'] = csv_df.index
        
        df = pd.concat([df, csv_df], ignore_index=True)
    
    return df


def preprocess_dataset(name='train'):
    print(f"Preprocessing {name} dataset...")
    df = load_dataset(f'./temp/{name}')

    df['SENTENCE'] = df['SENTENCE'].apply(avramandrei_clean_data)

    df = df.drop_duplicates(subset=['SENTENCE'], keep='first')  # NEW  # Some sentences are duplicated because of representation requirements of task2 and task3!

    if name != 'test':  # NEW
        df = df[df['SENTENCE'].str.len() > 30]  # 30 is a magic number found by doing some EDA over the dataset

    df.to_csv(f'./preprocessed/{name}.tsv', sep="\t", encoding="utf-8", header=True, index=False)


if __name__ == "__main__":
    convert_to_task1()  # Do this just once...

    os.system('mkdir "./preprocessed"')

    preprocess_dataset('train')
    preprocess_dataset('dev')
    preprocess_dataset('test')

    os.system('rmdir "./temp" /s /q ')
