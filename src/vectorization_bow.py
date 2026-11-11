import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')


def vectorize_dataset_bow(name, file_path, output_prefix):
    if not os.path.exists(file_path):
        return None

    df = pd.read_excel(file_path)
    text_col = 'Text_TFIDF' if 'Text_TFIDF' in df.columns else 'Clean Text'
    texts = np.array(df[text_col].fillna("đánh giá phim").astype(str).tolist())
    labels = np.array(df['Final Label'].astype(str).tolist())
    comment_ids = np.array(df['Comment ID'].tolist())

    train_texts, test_texts, train_labels, test_labels, train_ids, test_ids = train_test_split(
        texts, labels, comment_ids, test_size=0.2, stratify=labels, random_state=18
    )

    vec = CountVectorizer(ngram_range=(1, 2), min_df=2)
    X_train = vec.fit_transform(train_texts)
    X_test = vec.transform(test_texts)
    vocab = vec.get_feature_names_out()

    print(f"[{name} BoW] Total: {len(texts)} | Train: {X_train.shape} | Test: {X_test.shape} | Vocab: {len(vocab)}")

    output_dir = os.path.join("Data", "vectorized")
    os.makedirs(output_dir, exist_ok=True)

    data = {
        'name': name,
        'X_train': X_train,
        'y_train': train_labels,
        'train_texts': train_texts,
        'train_ids': train_ids,
        'X_test': X_test,
        'y_test': test_labels,
        'test_texts': test_texts,
        'test_ids': test_ids,
        'vectorizer': vec,
        'feature_names': vocab.tolist()
    }

    out_file = os.path.join(output_dir, f"{output_prefix}_bow_train_test.joblib")
    joblib.dump(data, out_file)

    if output_prefix == "tiktok":
        joblib.dump(data, os.path.join(output_dir, "bow_train_test.joblib"))


def main():
    vectorize_dataset_bow("TikTok", os.path.join("Data", "final_labeled_clean.xlsx"), "tiktok")
    vectorize_dataset_bow("Facebook", os.path.join("Data", "final_labeled_clean_v3.xlsx"), "facebook")


if __name__ == "__main__":
    main()
