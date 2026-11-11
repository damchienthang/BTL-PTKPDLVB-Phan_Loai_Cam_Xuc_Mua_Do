import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def main():
    path_fb = os.path.join("Data", "final_labeled_clean_v3.xlsx")
    path_tt = os.path.join("Data", "final_labeled_clean.xlsx")

    df_fb = pd.read_excel(path_fb)
    df_tt = pd.read_excel(path_tt)

    fb_text_col = 'Text_TFIDF' if 'Text_TFIDF' in df_fb.columns else 'Clean Text'
    tt_text_col = 'Text_TFIDF' if 'Text_TFIDF' in df_tt.columns else 'Clean Text'

    fb_texts = df_fb[fb_text_col].fillna("đánh giá phim").astype(str).tolist()
    tt_texts = df_tt[tt_text_col].fillna("đánh giá phim").astype(str).tolist()

    fb_labels = df_fb['Final Label'].astype(str)
    tt_labels = df_tt['Final Label'].astype(str)

    # 1. Thống kê số lượng & nhãn
    print(f"Facebook: {len(df_fb)} mẫu | Pos={sum(fb_labels=='Positive')} | Neu={sum(fb_labels=='Neutral')} | Neg={sum(fb_labels=='Negative')}")
    print(f"TikTok  : {len(df_tt)} mẫu | Pos={sum(tt_labels=='Positive')} | Neu={sum(tt_labels=='Neutral')} | Neg={sum(tt_labels=='Negative')}")

    # 2. Thống kê độ dài từ
    fb_lens = [len(t.split()) for t in fb_texts]
    tt_lens = [len(t.split()) for t in tt_texts]
    print(f"Độ dài TB: FB={np.mean(fb_lens):.1f} từ (Max {np.max(fb_lens)}) | TikTok={np.mean(tt_lens):.1f} từ (Max {np.max(tt_lens)})")

    # 3. Trích xuất đặc trưng TF-IDF
    vec_fb = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X_fb = vec_fb.fit_transform(fb_texts)
    vocab_fb = set(vec_fb.get_feature_names_out())

    vec_tt = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X_tt = vec_tt.fit_transform(tt_texts)
    vocab_tt = set(vec_tt.get_feature_names_out())

    common = vocab_fb & vocab_tt
    print(f"Từ vựng: FB={len(vocab_fb)} | TikTok={len(vocab_tt)} | Chung={len(common)} ({len(common)/len(vocab_fb|vocab_tt)*100:.1f}%)")

    # 4. Xuất biểu đồ so sánh
    output_dir = os.path.join("reports", "figures")
    os.makedirs(output_dir, exist_ok=True)

    # Biểu đồ 1: Phân bố nhãn
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels_list = ['Positive', 'Neutral', 'Negative']
    x = np.arange(len(labels_list))
    w = 0.35

    fb_p = [sum(fb_labels == l)/len(df_fb)*100 for l in labels_list]
    tt_p = [sum(tt_labels == l)/len(df_tt)*100 for l in labels_list]

    ax.bar(x - w/2, fb_p, w, label=f'Facebook ({len(df_fb)})', color='#3b5998')
    ax.bar(x + w/2, tt_p, w, label=f'TikTok ({len(df_tt)})', color='#00f2fe')
    ax.set_ylabel('Tỷ lệ (%)')
    ax.set_title('Phân Bố Nhãn Cảm Xúc: Facebook vs TikTok', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_list, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_label_distribution.png"), dpi=300)
    plt.close()

    # Biểu đồ 2: Top đặc trưng TF-IDF
    fb_scores = np.asarray(X_fb.mean(axis=0)).flatten()
    tt_scores = np.asarray(X_tt.mean(axis=0)).flatten()
    fb_words = vec_fb.get_feature_names_out()
    tt_words = vec_tt.get_feature_names_out()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Top 12 Đặc Trưng TF-IDF: Facebook vs TikTok', fontweight='bold')

    top_fb = fb_scores.argsort()[::-1][:12]
    ax1.barh(fb_words[top_fb][::-1], fb_scores[top_fb][::-1], color='#3b5998')
    ax1.set_title('Facebook')
    ax1.set_xlabel('Mean TF-IDF')
    ax1.grid(axis='x', linestyle='--', alpha=0.5)

    top_tt = tt_scores.argsort()[::-1][:12]
    ax2.barh(tt_words[top_tt][::-1], tt_scores[top_tt][::-1], color='#25F4EE')
    ax2.set_title('TikTok')
    ax2.set_xlabel('Mean TF-IDF')
    ax2.grid(axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_top_features_fb_vs_tt.png"), dpi=300)
    plt.close()

    print(f"Đã lưu các biểu đồ so sánh tại: {output_dir}")


if __name__ == "__main__":
    main()
