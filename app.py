import os
import streamlit as st

def menu():
    st.sidebar.page_link("app.py", label=" Trang chủ")
    st.sidebar.page_link("pages/Data_Analysis.py", label=" Phân tích dữ liệu")
    st.sidebar.page_link("pages/Input_Record.py", label=" Thêm dữ liệu")
    st.sidebar.page_link("pages/Prediction.py", label=" Dự đoán")

if __name__ == "__main__":
    st.set_page_config(
        page_title="Hệ thống Quản trị Đại học Chiến lược",
        layout="centered",
        page_icon="🎓",
    )

    # ====== MENU SIDEBAR ======
    menu()

    # ====== HEADER ======
    st.title(" University Strategy Predictor")
    st.markdown("---")
    
    st.header("Mục tiêu hệ thống")
    st.markdown("""
    Ứng dụng này được thiết kế để hỗ trợ các nhà quản lý giáo dục phân tích, đánh giá và dự báo 
    các chỉ số quan trọng trong bảng xếp hạng đại học thế giới (QS Rankings). 

    ### Các phân hệ chính:
    1. **Phân tích dữ liệu chuyên sâu**: Khám phá sự phân hóa điểm số, biến động của các chỉ số AR, ER, ISR... thông qua các biểu đồ trực quan.
    2. **Quản trị tập dữ liệu**: Cho phép cập nhật và bổ sung dữ liệu mới để làm giàu nguồn tri thức cho mô hình AI.
    3. **Dự báo Score Scaled**: Sử dụng sức mạnh của **Machine Learning** để dự đoán điểm chuẩn hóa dựa trên các kịch bản đầu tư vào các chỉ số thành phần.
    """)

    # ====== CÔNG NGHỆ ======
    st.subheader(" Nền tảng AI & Phân tích")
    st.markdown("""
    Hệ thống tích hợp các mô hình học máy hiện đại để đảm bảo độ chính xác trong dự báo:
    
    - **Linear Regression (Hồi quy tuyến tính)**: Phân tích mối tương quan đơn biến giữa Uy tín học thuật và Điểm chuẩn hóa.
    - **Random Forest Regression (Rừng ngẫu nhiên)**: Tổ hợp hàng trăm cây quyết định để xử lý dữ liệu đa biến, giúp dự báo **Score Scaled** với sai số thấp nhất (RMSE tối ưu).

    → **Lợi ích chiến lược**: 
    Giúp nhà quản trị xác định rõ "đòn bẩy" nào (Học thuật, Tuyển dụng hay Quốc tế hóa) mang lại hiệu quả thăng hạng nhanh nhất và bền vững nhất.
    """)

    # ====== CREDIT & DATA ======
    st.divider()
    st.subheader(" Nguồn dữ liệu & Công cụ")
    st.markdown(
        """
        - **Dữ liệu**: Dựa trên tập dữ liệu QS World University Rankings.
        - **Công nghệ**: Phát triển trên nền tảng Python, sử dụng thư viện **Streamlit** cho giao diện và **Scikit-learn** cho mô hình trí tuệ nhân tạo.
        """
    )

    st.info("Hãy sử dụng menu bên trái để bắt đầu quá trình phân tích dữ liệu.")