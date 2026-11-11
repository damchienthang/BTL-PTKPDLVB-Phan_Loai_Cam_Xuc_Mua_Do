import os
import sys
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Cấu hình font hiển thị tiếng Việt
plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def main():
    data_path = os.path.join("Data", "vectorized", "tfidf_train_test.joblib")
    if not os.path.exists(data_path):
        print(f"Không tìm thấy file: {data_path}. Vui lòng chạy vectorization.py trước!")
        return

    data = joblib.load(data_path)
    X_train = data['X_train']
    y_train = data['y_train']
    vocab = np.array(data['feature_names'])

    output_dir = os.path.join("reports", "figures")
    os.makedirs(output_dir, exist_ok=True)

    # 1. BIỂU ĐỒ 1: TOP 20 ĐẶC TRƯNG TF-IDF CAO NHẤT (TẬP TRAIN)
    mean_scores = np.asarray(X_train.mean(axis=0)).flatten()
    top20_idx = mean_scores.argsort()[::-1][:20]
    top20_words = vocab[top20_idx][::-1]
    top20_vals = mean_scores[top20_idx][::-1]

    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("Blues_d", n_colors=20)
    plt.barh(top20_words, top20_vals, color=colors)
    plt.title("Top 20 Đặc Trưng Có Trọng Số TF-IDF Trung Bình Cao Nhất (Tập Train)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Trọng số TF-IDF trung bình", fontsize=11)
    plt.ylabel("Đặc trưng (Từ / Cụm từ)", fontsize=11)
    plt.tight_layout()

    fig1_path = os.path.join(output_dir, "top_20_tfidf_overall.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Đã lưu biểu đồ 1: {fig1_path}")

    # 2. BIỂU ĐỒ 2: TOP 15 ĐẶC TRƯNG THEO TỪNG NHÃN CẢM XÚC
    labels_order = ["Negative", "Neutral", "Positive"]
    color_map = {
        "Negative": "#E74C3C",  # Đỏ
        "Neutral": "#95A5A6",   # Xám
        "Positive": "#2ECC71"   # Xanh lá
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle("Đặc Trưng TF-IDF Tiêu Biểu Cho Từng Nhãn Cảm Xúc", fontsize=14, fontweight='bold', y=1.02)

    for i, label in enumerate(labels_order):
        idx_mask = (y_train == label)
        if np.sum(idx_mask) == 0:
            continue

        mean_label_scores = np.asarray(X_train[idx_mask].mean(axis=0)).flatten()
        top15_idx = mean_label_scores.argsort()[::-1][:15]
        words = vocab[top15_idx][::-1]
        scores = mean_label_scores[top15_idx][::-1]

        axes[i].barh(words, scores, color=color_map[label])
        axes[i].set_title(f"Nhãn: {label.upper()} (Top 15)", fontsize=11, fontweight='bold')
        axes[i].set_xlabel("Mean TF-IDF", fontsize=9)
        axes[i].grid(axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout()
    fig2_path = os.path.join(output_dir, "top_features_by_sentiment.png")
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Đã lưu biểu đồ 2: {fig2_path}")


if __name__ == "__main__":
    main()

