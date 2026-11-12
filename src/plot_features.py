import os
import sys
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def main():
    data_path = os.path.join("Data", "vectorized", "tfidf_train_test.joblib")
    if not os.path.exists(data_path):
        return

    data = joblib.load(data_path)
    X_train = data['X_train']
    y_train = data['y_train']
    vocab = np.array(data['feature_names'])

    output_dir = os.path.join("reports", "figures")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Top 20 đặc trưng tập Train
    mean_scores = np.asarray(X_train.mean(axis=0)).flatten()
    top20_idx = mean_scores.argsort()[::-1][:20]
    words = vocab[top20_idx][::-1]
    scores = mean_scores[top20_idx][::-1]

    plt.figure(figsize=(9, 5.5))
    plt.barh(words, scores, color=sns.color_palette("Blues_d", n_colors=20))
    plt.title("Top 20 Đặc Trưng TF-IDF (Tập Train)", fontweight='bold')
    plt.xlabel("Mean TF-IDF")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_20_tfidf_overall.png"), dpi=300)
    plt.close()

    # 2. Top đặc trưng theo nhãn cảm xúc
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("Đặc Trưng TF-IDF Tiêu Biểu Theo Nhãn Cảm Xúc (Phim Trại Buôn Người)", fontweight='bold')

    order = [("Negative", "#E74C3C"), ("Neutral", "#95A5A6"), ("Positive", "#2ECC71")]
    for i, (label, color) in enumerate(order):
        mask = (y_train == label)
        if np.sum(mask) == 0:
            continue
        scores_lbl = np.asarray(X_train[mask].mean(axis=0)).flatten()
        top_idx = scores_lbl.argsort()[::-1][:15]
        axes[i].barh(vocab[top_idx][::-1], scores_lbl[top_idx][::-1], color=color)
        axes[i].set_title(f"Nhãn: {label} (Top 15)", fontweight='bold')
        axes[i].grid(axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_features_by_sentiment.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Đã lưu biểu đồ tại: {output_dir}")


if __name__ == "__main__":
    main()
