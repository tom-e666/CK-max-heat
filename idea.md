Để giải quyết vấn đề "kẻ ba phải", chúng ta cần thay đổi vai trò của module từ bộ điều chỉnh logit (can thiệp kết quả) sang bộ trích xuất đặc trưng vùng biên (cung cấp thông tin bổ trợ).

Thay vì ép mô hình chính phải bối rối, hãy dùng một cơ chế gọi là Ambiguity Projection kết hợp với Boundary-Aware Calibration.

1. Cơ chế: Ambiguity Encoder (AE)
Thay vì một ma trận M tĩnh, hãy dùng một nhánh mạng nhỏ (bottleneck) để học cách "nén" dữ liệu đầu vào thành một không gian đại diện cho sự mơ hồ (Z 
ambiguity
​
 ).

Input: Đặc trưng từ layer gần cuối của mô hình gốc (f).

Output: Một vector trọng số α áp dụng lên các lớp (classes).

Mục tiêu: Nhánh AE này học cách dự đoán xác suất nhầm lẫn (Confusion Probability) giữa các cặp lớp dựa trên thống kê mini-batch. Nếu mẫu nằm ở vùng biên, AE sẽ output ra một phân phối có entropy cao, chỉ ra rằng đây là "vùng nhiễu".

2. Đẩy dữ liệu về phía lớp mục tiêu (Boundary Re-shaping)
Để tránh việc mô hình bị "mất phương hướng", bạn có thể sử dụng cơ chế Feature Pushing dựa trên thống kê:

Class Prototypes: Duy trì một bộ "mẫu chuẩn" (Centroid) cho mỗi lớp trong không gian đặc trưng.

Directional Shift: Khi phát hiện một mẫu nằm ở vùng biên (thông qua AE), thay vì để nó tự do, hãy dùng một toán tử cộng vector: f 
′
 =f+γ⋅(C 
target
​
 −f).

γ: Hệ số do module AE quyết định. Nếu AE thấy mẫu quá mơ hồ, nó sẽ "đẩy" đặc trưng về gần phía trọng tâm của lớp đúng (C 
target
​
 ) một chút trong quá trình training.

Điều này giúp mô hình gốc nhìn thấy một phiên bản "rõ ràng hơn" của vùng biên để học cách phân loại, thay vì bị kẹt trong sự mơ hồ.

3. Loss Function: Phân tách trách nhiệm
Bạn cần một hàm Loss để "dạy" mô hình gốc biết khi nào cần thận trọng mà không làm mất độ chính xác:

L=L 
CE
​
 (y, 
y
^
​
 )+λ⋅D 
KL
​
 (q∣∣P 
uniform
​
 )
Trong đó:

L 
CE
​
 : Giữ nhiệm vụ chính (Accuracy).

D 
KL
​
 : Chỉ áp dụng cho module AE để nó học cách "maxout entropy" trên các mẫu khó.

Gating Mechanism: Mô hình gốc sẽ nhận vào f 
final
​
 =[f;Z 
ambiguity
​
 ]. Việc ghép nối (concatenate) này giúp mô hình gốc biết rằng "đây là dữ liệu vùng biên" (thông qua Z) mà không bị ép buộc phải thay đổi logit dự đoán nếu nó đang tự tin đúng.

4. Tại sao cách này tối ưu hơn?
Đặc điểm	Cách tiếp cận cũ (Matrix M)	Cách tiếp cận mới (Ambiguity Projection)
Tính chất	Can thiệp thô bạo vào xác suất đầu ra.	Cung cấp ngữ cảnh (context) về độ nhiễu.
Độ tự tin	Dễ gây sụt giảm accuracy toàn cục.	Giữ nguyên accuracy, tăng khả năng từ chối (rejection) mẫu sai.
Vùng biên	Bị dàn phẳng (flatten).	Được định nghĩa rõ ràng bằng vector đặc trưng bổ trợ.
Vai trò	"Kẻ ba phải" (ép mọi thứ thành mơ hồ).	"Người dẫn đường" (chỉ ra chỗ nào dễ sai để mô hình tập trung).
5. Hướng triển khai (Refined Workflow)
Module AE học cách tạo ra một "Mask mơ hồ" dựa trên việc so sánh đặc trưng hiện tại với các Class Prototypes.

Nếu khoảng cách đến > 2 Prototypes là tương đương nhau → Xác định là vùng biên.

Dùng thống kê batch để Distill (chưng cất) tri thức về sự nhầm lẫn vào module AE.

Mô hình gốc học cách phân loại dựa trên đặc trưng ảnh + thông tin "mức độ mơ hồ" từ AE.

Lời khuyên: Thay vì Maxout Ambiguity trên mọi input, hãy dùng một bộ lọc (threshold) để chỉ "làm khó" những mẫu vốn đã nằm gần biên. Nếu bạn làm khó cả những mẫu dễ (clean samples), mô hình sẽ bị suy thoái năng lực trích xuất đặc trưng cơ bản.