import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from nlpaug.augmenter.word import ContextualWordEmbsAug
from nlpaug.flow import Sometimes

# import torch
# from transformers import PegasusForConditionalGeneration, PegasusTokenizer

# model_name = 'tuner007/pegasus_paraphrase'
# torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
# tokenizer = PegasusTokenizer.from_pretrained(model_name)
# model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

# print(torch_device)


# def get_response(input_text, num_return_sequences, num_beams):
#     try:
#         batch = tokenizer([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)

#         translated = model.generate(**batch,max_length=60,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)

#         tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
#     except:
#         print("ERROR: " + input_text)
#         return []
#     return tgt_text


# def augment_dataset(name='train', num_paraphrases=1):
#     print(f"Augmenting {name} dataset...")
#     df = pd.read_csv(f'./{name}.tsv', sep="\t", header=0, encoding='utf-8',
#                      dtype={"SENTENCE": str, "HAS_DEF": np.uint8, 'SOURCE': str, 'INDEX': np.uint16})

#     augmented_sentences = []
#     for _, row in tqdm(df.iterrows(), total=df.shape[0]):
#         if pd.isna(row['SENTENCE']) or len(row['SENTENCE']) <= 10:  # Filter too-short sentences
#             continue

#          # Keep the original sentence
#         augmented_sentences.append({'SENTENCE': row['SENTENCE'], 'HAS_DEF': row['HAS_DEF'], 'SOURCE': row['SOURCE'], 'INDEX': row['INDEX']})

#         paraphrases = get_response(row['SENTENCE'], num_paraphrases, 10)  # Generate paraphrases

#         # Add paraphrases
#         new_rows = [{'SENTENCE': p, 'HAS_DEF': row['HAS_DEF'], 'SOURCE': row['SOURCE'], 'INDEX': row['INDEX']} for p in paraphrases]
#         augmented_sentences.extend(new_rows)

#     augmented_df = pd.DataFrame(augmented_sentences)
#     augmented_df.to_csv(f'./augmented/{name}.tsv', sep="\t", encoding="utf-8", header=True, index=False)

def augment_dataset(name='train'):
    print(f"Augmenting {name} dataset...")
    df = pd.read_csv(f'./preprocessed/{name}.tsv', sep="\t", header=0, encoding='utf-8',
                     dtype={"SENTENCE": str, "HAS_DEF": np.uint8, 'SOURCE': str, 'INDEX': np.uint16})

    df['SYNTHETIC'] = 0

    augmenter = ContextualWordEmbsAug(
        "roberta-base", "roberta", action="substitute", device="cuda", aug_min=1, aug_max=4, aug_p=0.2, stopwords=['<link>', '<equation>', '(<link)'])

    augmented_rows = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        augmented_sentence = augmenter.augment(row['SENTENCE'], n=1)[0]
        augmented_rows.append({'SENTENCE': augmented_sentence,
                              'HAS_DEF': row['HAS_DEF'], 'SOURCE': row['SOURCE'], 'INDEX': row['INDEX'], 'SYNTHETIC': 1})

    augmented_df = pd.DataFrame(augmented_rows)

    augmented_df = pd.concat([df, augmented_df], ignore_index=True)

    augmented_df.to_csv(
        f'./augmented/{name}.tsv', sep="\t", encoding="utf-8", header=True, index=False)


if __name__ == "__main__":
    os.system('mkdir "./augmented"')

    augment_dataset('train')
    augment_dataset('dev')
    # Never modify the test set!
