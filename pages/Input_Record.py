import streamlit as st
import pandas as pd

def menu():
    st.sidebar.page_link("app.py", label=" Trang chủ")
    st.sidebar.page_link("pages/Data_Analysis.py", label=" Phân tích dữ liệu")
    st.sidebar.page_link("pages/Input_Record.py", label=" Thêm dữ liệu")
    st.sidebar.page_link("pages/Prediction.py", label=" Dự đoán")

st.set_page_config(page_title="Thêm dữ liệu", layout="wide")
st.title("Thêm dữ liệu mới vào hệ thống")

menu()

# Load data
try:
    df = pd.read_csv("UNI.csv")
    # Chuẩn hóa tên cột về chữ thường để tránh lỗi khi lưu
    df.columns = [col.strip().lower() for col in df.columns]
except FileNotFoundError:
    st.error("Không tìm thấy file UNI.csv!")
    st.stop()

st.subheader("Nhập thông tin chi tiết của trường")

# Chia layout để nhập thông tin cơ bản
col_info1, col_info2 = st.columns(2)
with col_info1:
    institution = st.text_input("Tên trường (Institution)")
with col_info2:
    location = st.text_input("Quốc gia (Location)")

st.markdown("---")
st.write("**Nhập điểm các chỉ số thành phần:**")

# Chia cột để nhập điểm số cho gọn gàng
col1, col2, col3 = st.columns(3)

with col1:
    ar = st.number_input("AR Score (Academic Reputation - Danh tiếng học thuật)", 0.0, 100.0, 50.0)
    er = st.number_input("ER Score (Employer Reputation - Uy tín nhà tuyển dụng)", 0.0, 100.0, 50.0)

with col2:
    ifr = st.number_input("IFR Score (International Faculty Ratio - Tỷ lệ giảng viên quốc tế)", 0.0, 100.0, 50.0)
    isr = st.number_input("ISR Score (International Student Ratio - Tỷ lệ sinh viên quốc tế)", 0.0, 100.0, 50.0)

with col3:
    irn = st.number_input("IRN Score (International Research Network - Mạng lưới nghiên cứu quốc tế)", 0.0, 100.0, 50.0)
    # Thêm Score Scaled nếu cần thiết để khớp với mô hình dự báo
    ss = st.number_input("Score Scaled (Điểm chuẩn hóa tổng thể)", 0.0, 100.0, 50.0)

st.markdown("---")

if st.button("Lưu dữ liệu vào hệ thống"):
    if institution.strip() == "" or location.strip() == "":
        st.warning("Vui lòng nhập đầy đủ Tên trường và Quốc gia.")
    else:
        new_row = {
            "institution": institution,
            "location": location,
            "ar score": ar,
            "er score": er,
            "ifr score": ifr,
            "isr score": isr,
            "irn score": irn,
            "score scaled": ss
        }

        # Thêm hàng mới vào dataframe
        new_df = pd.DataFrame([new_row])
        df = pd.concat([df, new_df], ignore_index=True)
        
        # Lưu lại file CSV
        df.to_csv("UNI.csv", index=False)

        st.success(f"Đã lưu thành công dữ liệu cho trường: {institution}")
        st.balloons()

# Hiển thị dữ liệu kiểm tra
st.subheader("Danh sách dữ liệu mới cập nhật (10 bản ghi cuối)")
st.dataframe(df.tail(10), use_container_width=True)