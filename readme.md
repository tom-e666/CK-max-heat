# Latent Ambiguity Shaper (LAS)

LAS là một ý tưởng huấn luyện giúp mô hình nhận diện cảm xúc học tốt hơn ở vùng ranh giới giữa các lớp (nơi mô hình dễ nhầm lẫn và quá tự tin sai).

## 1. Vấn đề
Trong bài toán emotion recognition, nhiều mẫu nằm gần biên quyết định giữa 2-3 lớp. Nếu chỉ tối ưu Cross-Entropy chuẩn, mô hình thường:
- Đúng trên mẫu dễ nhưng kém ổn định trên mẫu khó.
- Overconfident ở vùng biên.
- Dễ tạo lỗi nhầm cặp lớp giống nhau.

## 2. Ý tưởng LAS
Thay vì can thiệp thô vào xác suất đầu ra, LAS bổ sung một nhánh học mơ hồ tiềm ẩn (Ambiguity Encoder) và cơ chế đẩy đặc trưng vùng biên về phía centroid lớp đúng.

Thành phần chính:
- Backbone CNN tạo embedding đặc trưng f.
- Ambiguity Encoder tạo phân phối mơ hồ q trên các lớp.
- Gamma head ước lượng độ mạnh cần đẩy mẫu vùng biên.
- Classifier nhận đặc trưng ghép [f; q] để dự đoán cuối.

## 3. Cơ chế huấn luyện
1. Duy trì prototype (centroid) cho từng lớp trong không gian embedding.
2. Phát hiện mẫu vùng biên bằng chênh lệch khoảng cách giữa centroid lớp thật và centroid lớp đối thủ gần nhất.
3. Với mẫu vùng biên, áp dụng Feature Pushing:

	f' = f + gamma * push_scale * (C_target - f)

4. Tối ưu tổng loss:

	L = CE_clean + lambda_push * CE_boundary_pushed + lambda_ae * KL(q || Uniform)

Ý nghĩa:
- CE_clean giữ accuracy lõi.
- CE_boundary_pushed làm biên quyết định rõ hơn.
- KL với Uniform buộc nhánh ambiguity biểu diễn đúng tính mơ hồ ở mẫu khó (entropy cao).

## 4. Triển khai trong repo
- Kiến trúc model LAS: [model.py](model.py)
- Vòng train + prototype + boundary pushing: [train.py](train.py)
- Ghi chú ý tưởng chi tiết: [idea.md](idea.md)
- Dữ liệu CK+ dạng CSV: [ckextended.csv](ckextended.csv)

## 5. Chạy huấn luyện
Lệnh gợi ý:

	 python.exe train.py --device cuda --epochs 30 --batch-size 16 --num-workers 4 --lr 0.001 --embedding-dim 128 --proto-momentum 0.95 --boundary-margin 0.15 --push-scale 0.5 --lambda-ae 0.02 --lambda-push 0.25

## 6. Kỳ vọng kết quả
- Tăng độ ổn định ở các cặp lớp dễ nhầm.
- Giảm overconfidence ở vùng biên.
- Cải thiện confusion matrix theo hướng tập trung lỗi ít hơn vào các lớp gần nhau.

## 7. Gợi ý ablation
- Baseline CNN (không LAS).
- LAS không pushing (chỉ ambiguity).
- LAS đầy đủ (ambiguity + pushing + prototype boundary).

Đo lường:
- Validation accuracy.
- Per-class F1.
- Confusion matrix.