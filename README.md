# 💳 Credit Card Customer Churn Prediction with Machine Learning

## 📌 Tổng quan dự án (Project Overview)
Dự án tập trung vào việc xây dựng hệ thống dự báo rủi ro rời bỏ của khách hàng thẻ tín dụng cho ngân hàng. Việc nhận diện sớm giúp ngân hàng tối ưu hóa chi phí giữ chân khách hàng (tiết kiệm gấp 5-25 lần so với tìm khách hàng mới).

## 🛠 Phương pháp nghiên cứu (Methodology)
Dự án áp dụng quy trình phân tích dữ liệu chuyên sâu:
1. **Tiền xử lý:** Làm sạch dữ liệu, xử lý outliers và chuẩn hóa dữ liệu bằng `StandardScaler`.
2. **Kỹ thuật đặc trưng (Feature Engineering):** Tạo các biến mới như `Avg_Trans_Value` và `Customer_Segment`.
3. **Phân cụm (K-Means):** Phân loại hành vi khách hàng thành các nhóm (VIP, Ngủ đông, Áp lực nợ).
4. **Giảm chiều dữ liệu (PCA):** Xử lý hiện tượng đa cộng tuyến, nén 20+ biến xuống 4 thành phần chính (PCs).
5. **Huấn luyện mô hình:** So sánh và tối ưu hóa **Random Forest** và **Logistic Regression (Lasso)** qua Grid Search CV.

## 📊 Kết quả ấn tượng (Key Results)
- **Độ chính xác toàn cục:** 95.76%
- **Chỉ số ROC-AUC:** **0.99** (Khả năng phân loại gần như lý tưởng)
- **Recall (Nhóm rủi ro):** **79.3%** (Bắt trọn 8/10 khách hàng thực sự có ý định rời bỏ).
- **Insight quan trọng:** Tần suất giao dịch (`Total_Trans_Ct`) là chỉ báo sớm quan trọng nhất.

## 🚀 Công cụ sử dụng
- **Ngôn ngữ:** Python (Pandas, NumPy, Scikit-learn)
- **Trực quan hóa:** Matplotlib, Seaborn
- **Thuật toán:** Random Forest, PCA, K-Means, Lasso Regularization.

## 📂 Cấu trúc thư mục
- `Code/`: Chứa file Jupyter Notebook xử lý chi tiết.
- `Data/`: Bộ dữ liệu từ Kaggle (Credit Card Customers).
