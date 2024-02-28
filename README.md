# Final-Project-Applied-Data-Science

# **Giai đoạn 1: Giới thiệu đồ án**
## **1.1. Đề tài**
- Đề tài: Dự đoán doanh thu của một bộ phim.
- Mục tiêu:
  - Dự đoán được doanh thu của một bộ phim với những thuộc tính được biết trước của bộ phim đó.
  - Đồng thời cố gắng đạt được kết quả cao khi thi trên **`Kaggle`**.
- Ý nghĩa thực tiễn: Giúp các nhà sản xuất dự đoán được doanh thu bộ phim, tìm ra được những yếu tố giúp mang đến doanh thu cao cho một bộ phim từ đó khi sản xuất những bộ phim tiếp theo sẽ mang lại daonh thu tốt nhất.
## **1.2. Bộ dữ liệu**
- Giới thiệu bộ dữ liệu:
  - Có 2 phần chính là tập **`train`** và tập **`test`**.
  - Tập **`train`** có **`3000`** mẫu và **`23`** thuộc tính.
  - Tập **`test`** có **`4000`** mẫu và **`22`** thuộc tính.
- Link bộ dữ liệu được lấy trên Kaggle: [Xem tại đây](https://www.kaggle.com/competitions/tmdb-box-office-prediction)

# **Giai đoạn 2: Khai thác thông tin cơ bản của dữ liệu**
## **2.1. Chuẩn bị dữ liệu**
## **2.2. Phân tích - Khám phá Dữ liệu**
- Cấu trúc bộ dữ liệu.
- Chất lượng bộ dữ liệu.
- Nội dung bộ dữ liệu.
  - Phân tích đơn biến.
  - Phân tích hai biến.
  - Phân tích đa biến.

# **Giai đoạn 3: Trích chọn đặc trưng (Feature Selection), Rút trích đặc trưng (Feature Extraction), Khai thác dữ liệu (Data mining), Thực nghiệm và đánh giá kết quả**
## **3.1. Trích chọn đặc trưng và Rút trích đặc trưng**
- Tình trạng khuyết dữ liệu (Missing data).
- Số hóa giá trị thuộc tính và Xử lí tình trạng khuyết dữ liệu (Missing data).
- Giảm chiều dữ liệu.
## **3.2. Khai thác dữ liệu, Thực nghiệm và đánh giá kết quả**
- Áp dụng mô hình học máy:
  - Do dữ liệu không có  nhiều nên ta sẽ dùng phương pháp k-fold cross validation để tăng độ hiệu quả của mô hình.
  - Những mô hình mà nhóm chọn từ những mô hình từ đơn giản đến phức tạp: **`Linear Regression`**, **`Random Forest`**, **`Xgboost`**, **`Lightgbm`**, **`CatBoost`** để giải quyết bài toán.
  - Lý do lựa chọn các thuật toán trên:
    + Đây đều là những thuật toán đều có thể áp dụng cho bài toán dự đoán doanh thu phim.
    + Thời gian giải quyết bài toán nhanh.
    + Hiệu suất tốt và khả năng ứng dụng linh hoạt.
  - Độ đo sử dụng để đánh giá kết quả khai thác: root mean square error.
  - Lý do sử dụng độ đo **`RMSE`**:
    + **`RMSE`** thường được sử dụng trong các bài toán dự đoán số, và doanh thu phim có tính số học rõ ràng.
    + **`RMSE`** có cùng đơn vị với đại lượng đang được dự đoán. Điều này giúp việc đánh giá hiệu suất dễ dàng và có ý nghĩa thực tiễn, vì chúng ta có thể so sánh giá trị **`RMSE`** trực tiếp với đại lượng thực tế mà chúng ta quan tâm.
    + Trong bài toán dự đoán doanh thu phim, nếu có những điểm dữ liệu quan trọng mà mô hình không dự đoán chính xác, **`RMSE`** sẽ tăng đáng kể, và điều này cho phép chúng ta nhận ra sự nhạy cảm và cần thiết để cải thiện mô hình.
      
- Đánh giá kết quả:
  - Cả 5 thuật toán đều mô hình hóa thành công dữ liệu.
  - Từ biểu đồ cho thấy các thuộc tính **`ratio_budget_year`**, **`log_budget`**, **`release_year`**, **`popularity_to_mean_year`** là những yếu tố cực kỳ quan trọng khi xuất hiện trong 3 trên 4 mô hình.
  - Mô hình đưa ra độ đo hiệu quả là: **`Random Forest`**, **`Xgboost`**, **`Lightgbm`**, **`CatBoost`** trong đó hiệu quả nhất là mô hình **`CatBoost`**, tuy nhiên chênh lệch là không đáng kể.
  - Thời gian chạy mô hình nhanh nhất là **`Linear Regression`**, mất chưa tới 1 phút để đưa ra kết quả, tuy nhiên kết quả cho ra lại là kết quả tệ nhất, chứng tỏ mô hình **`Linear Regression`** không phù hợp với bài toán này.

- Kết quả trên **`Kaggle`**:
  - Ta sử dụng mô hình **`CatBoost`** để áp dụng và nộp.
  - Kết quả: **`1.79145`**.
  - Với kết quả như trên thì thứ hạng mà nhóm có thể đạt được là 302 trong trên tổng số 1395 người tham gia.
   
# **4. Hướng dẫn chạy code**
- Tải file notebook lên Google Colab và chạy.
