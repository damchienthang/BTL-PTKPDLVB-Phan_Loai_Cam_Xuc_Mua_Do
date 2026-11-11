import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def main():
    print("=" * 75)
    print(" BÁO CÁO ĐỐI SÁNH DỮ LIỆU & ĐẶC TRƯNG: FACEBOOK vs TIKTOK")
    print("=" * 75)

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

    # -------------------------------------------------------------
    # 1. SO SÁNH QUY MÔ & PHÂN BỐ NHÃN
    # -------------------------------------------------------------
    print("\n[1] SO SÁNH QUY MÔ & PHÂN BỐ NHÃN CẢM XÚC:")
    print(f"{'Tiêu chí':<25} | {'Facebook (v3)':<20} | {'TikTok (Mới)':<20}")
    print("-" * 72)
    print(f"{'Tổng số bình luận':<25} | {len(df_fb):<20,d} | {len(df_tt):<20,d}")
    
    for lbl in ['Positive', 'Neutral', 'Negative']:
        fb_c = sum(fb_labels == lbl)
        tt_c = sum(tt_labels == lbl)
        fb_pct = fb_c / len(df_fb) * 100
        tt_pct = tt_c / len(df_tt) * 100
        print(f"  + Nhãn {lbl:<16} | {fb_c:4d} ({fb_pct:.2f}%)       | {tt_c:4d} ({tt_pct:.2f}%)")

    # -------------------------------------------------------------
    # 2. SO SÁNH ĐỘ DÀI BÌNH LUẬN (SỐ TỪ)
    # -------------------------------------------------------------
    fb_lens = [len(t.split()) for t in fb_texts]
    tt_lens = [len(t.split()) for t in tt_texts]

    print("\n[2] SO SÁNH ĐỘ DÀI BÌNH LUẬN (SỐ TỪ/CÂU):")
    print(f"{'Chỉ số':<25} | {'Facebook':<20} | {'TikTok':<20}")
    print("-" * 72)
    print(f"{'Độ dài trung bình':<25} | {np.mean(fb_lens):<20.2f} | {np.mean(tt_lens):<20.2f}")
    print(f"{'Độ lệch chuẩn (Std)':<25} | {np.std(fb_lens):<20.2f} | {np.std(tt_lens):<20.2f}")
    print(f"{'Độ dài ngắn nhất (Min)':<25} | {np.min(fb_lens):<20d} | {np.min(tt_lens):<20d}")
    print(f"{'Độ dài dài nhất (Max)':<25} | {np.max(fb_lens):<20d} | {np.max(tt_lens):<20d}")

    # -------------------------------------------------------------
    # 3. SO SÁNH KHÔNG GIAN TỪ VỰNG & TF-IDF
    # -------------------------------------------------------------
    vec_fb = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X_fb = vec_fb.fit_transform(fb_texts)
    vocab_fb = set(vec_fb.get_feature_names_out())

    vec_tt = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X_tt = vec_tt.fit_transform(tt_texts)
    vocab_tt = set(vec_tt.get_feature_names_out())

    common_vocab = vocab_fb.intersection(vocab_tt)
    fb_only = vocab_fb - vocab_tt
    tt_only = vocab_tt - vocab_fb

    print("\n[3] SO SÁNH KHÔNG GIAN TỪ VỰNG N-GRAM (TF-IDF):")
    print(f"  * Tổng từ vựng Facebook (min_df=2) : {len(vocab_fb):,} n-grams")
    print(f"  * Tổng từ vựng TikTok (min_df=2)   : {len(vocab_tt):,} n-grams")
    print(f"  * Từ vựng GIAO THOA (chung cả 2)  : {len(common_vocab):,} ({len(common_vocab)/len(vocab_fb | vocab_tt)*100:.1f}%)")
    print(f"  * Từ vựng ĐẶC THÙ chỉ có ở Facebook: {len(fb_only):,}")
    print(f"  * Từ vựng ĐẶC THÙ chỉ có ở TikTok  : {len(tt_only):,}")

    # -------------------------------------------------------------
    # 4. TOP ĐẶC TRƯNG TIÊU BIỂU CỦA TỪNG NỀN TẢNG
    # -------------------------------------------------------------
    fb_scores = np.asarray(X_fb.mean(axis=0)).flatten()
    tt_scores = np.asarray(X_tt.mean(axis=0)).flatten()
    fb_words = vec_fb.get_feature_names_out()
    tt_words = vec_tt.get_feature_names_out()

    print("\n[4] TOP 10 ĐẶC TRƯNG CÓ TRỌNG SỐ TF-IDF CAO NHẤT:")
    print(f"{'Hạng':<5} | {'Facebook':<25} | {'TikTok':<25}")
    print("-" * 60)
    top_fb_idx = fb_scores.argsort()[::-1][:10]
    top_tt_idx = tt_scores.argsort()[::-1][:10]
    for r in range(10):
        w_fb = f"{fb_words[top_fb_idx[r]]} ({fb_scores[top_fb_idx[r]]:.4f})"
        w_tt = f"{tt_words[top_tt_idx[r]]} ({tt_scores[top_tt_idx[r]]:.4f})"
        print(f" {r+1:2d}   | {w_fb:<25} | {w_tt:<25}")

    # -------------------------------------------------------------
    # 5. XUẤT CÁC BIỂU ĐỒ ĐỐI SÁNH
    # -------------------------------------------------------------
    output_dir = os.path.join("reports", "figures")
    os.makedirs(output_dir, exist_ok=True)

    # Biểu đồ 1: So sánh phân bố nhãn
    fig, ax = plt.subplots(figsize=(8, 5))
    labels_list = ['Positive', 'Neutral', 'Negative']
    x = np.arange(len(labels_list))
    width = 0.35

    fb_pcts = [sum(fb_labels == l)/len(df_fb)*100 for l in labels_list]
    tt_pcts = [sum(tt_labels == l)/len(df_tt)*100 for l in labels_list]

    b1 = ax.bar(x - width/2, fb_pcts, width, label='Facebook (2.087 mẫu)', color='#3b5998')
    b2 = ax.bar(x + width/2, tt_pcts, width, label='TikTok (1.307 mẫu)', color='#00f2fe')

    ax.set_ylabel('Tỷ lệ (%)', fontsize=11)
    ax.set_title('So Sánh Phân Bố Nhãn Cảm Xúc: Facebook vs TikTok', fontsize=13, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_list, fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in b1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontsize=9)
    for bar in b2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    fig1_path = os.path.join(output_dir, "comparison_label_distribution.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"\n[5] Đã lưu biểu đồ 1: {fig1_path}")

    # Biểu đồ 2: So sánh Top 12 đặc trưng TF-IDF
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('So Sánh Top 12 Đặc Trưng TF-IDF: Facebook vs TikTok', fontsize=14, fontweight='bold', y=0.98)

    top12_fb_idx = fb_scores.argsort()[::-1][:12]
    ax1.barh(fb_words[top12_fb_idx][::-1], fb_scores[top12_fb_idx][::-1], color='#3b5998')
    ax1.set_title('Facebook (Bình luận phim)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Trọng số TF-IDF trung bình', fontsize=10)
    ax1.grid(axis='x', linestyle='--', alpha=0.5)

    top12_tt_idx = tt_scores.argsort()[::-1][:12]
    ax2.barh(tt_words[top12_tt_idx][::-1], tt_scores[top12_tt_idx][::-1], color='#25F4EE')
    ax2.set_title('TikTok (Bình luận phim)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Trọng số TF-IDF trung bình', fontsize=10)
    ax2.grid(axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout()
    fig2_path = os.path.join(output_dir, "comparison_top_features_fb_vs_tt.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"    Đã lưu biểu đồ 2: {fig2_path}")


if __name__ == "__main__":
    main()
