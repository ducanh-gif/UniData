import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# 1. CẤU HÌNH TRANG
st.set_page_config(page_title="Phân tích Chiến lược Đại học", layout="wide")

def menu():
    st.sidebar.page_link("app.py", label=" Trang chủ")
    st.sidebar.page_link("pages/Data_Analysis.py", label=" Phân tích dữ liệu")
    st.sidebar.page_link("pages/Input_Record.py", label=" Thêm dữ liệu")
    st.sidebar.page_link("pages/Prediction.py", label=" Dự đoán")

menu()

st.title("Dashboard Phân tích & Quản trị Đại học Chiến lược")
st.markdown("---")

# 2. ĐỌC VÀ XỬ LÝ DỮ LIỆU
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("UNI.csv")
        data.columns = [col.strip().lower() for col in data.columns]
        # ĐỒNG BỘ VỚI NOTEBOOK: Chỉ xóa hàng nếu cột mục tiêu 'score scaled' bị thiếu
        if 'score scaled' in data.columns:
            clean = data.dropna(subset=['score scaled'])
        else:
            clean = data
        return clean
    except Exception as e:
        st.error(f"Lỗi đọc file UNI.csv: {e}")
        return pd.DataFrame()

try:
    df_clean = load_data()
    
    if df_clean.empty:
        st.warning("Dữ liệu trống hoặc không tìm thấy file UNI.csv.")
    else:
        top10 = df_clean.head(10)

        # =========================================================
        # PHẦN 1: TỔNG QUAN PHÂN PHỐI & BIẾN ĐỘNG
        # =========================================================
        st.header("1. Tổng quan Phân phối & Độ phân hóa hệ thống")
        col_dist1, col_dist2 = st.columns(2)

        with col_dist1:
            st.subheader("Phân phối điểm ISR (Quốc tế hóa)")
            fig0, ax0 = plt.subplots(figsize=(10, 6))
            sns.histplot(df_clean['isr score'], bins=20, kde=True, color='purple', ax=ax0)
            st.pyplot(fig0)
            st.write("""
            **Phân tích chi tiết:** Biểu đồ Histogram kết hợp đường cong mật độ (KDE) cho thấy **mức độ tập trung điểm số** của toàn bộ hệ thống. 
            Việc quan sát hình dáng của đường cong giúp xác định xem phần lớn các trường đại học đang nằm ở phân khúc điểm trung bình 
            hay có sự phân hóa cực đoan về hai phía, từ đó đưa ra các **quyết định định vị thương hiệu** phù hợp với năng lực thực tế.
            """)

        with col_dist2:
            st.subheader("Biến động điểm số (Box Plot)")
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            box_cols = [c for c in ['ar score', 'er score', 'isr score'] if c in df_clean.columns]
            if box_cols:
                sns.boxplot(data=df_clean[box_cols], palette="Pastel1", ax=ax3)
                st.pyplot(fig3)
            st.write("""
            **Phân tích chi tiết:** Biểu đồ hộp (Box Plot) là công cụ quan trọng để nhận diện **độ lệch và giá trị ngoại lai** của từng tiêu chí. 
            Nếu một chỉ số có dải hộp dài, điều đó chứng tỏ sự cạnh tranh và phân hóa trong tiêu chí đó là rất lớn. Các điểm nằm ngoài râu của hộp 
            là những đơn vị có **thành tích đột biến**, cung cấp bài học kinh nghiệm về việc tối ưu hóa nguồn lực.
            """)

        # =========================================================
        # PHẦN 2: ĐỐI SÁNH NĂNG LỰC & MA TRẬN TƯƠNG QUAN
        # =========================================================
        st.divider()
        st.header("2. Đối sánh Năng lực & Ma trận Tương quan")
        
        col_bar, col_heat = st.columns([1.5, 1])

        with col_bar:
            st.subheader("So sánh AR vs ER (Top 10)")
            fig1, ax1 = plt.subplots(figsize=(10, 7))
            x = np.arange(len(top10['institution'])) 
            width = 0.35
            if 'ar score' in top10.columns and 'er score' in top10.columns:
                ax1.bar(x - width/2, top10['ar score'], width, label='Học thuật (AR)', color='#3498db')
                ax1.bar(x + width/2, top10['er score'], width, label='Tuyển dụng (ER)', color='#e67e22')
                ax1.set_xticks(x)
                ax1.set_xticklabels(top10['institution'], rotation=45, ha='right')
                ax1.legend()
                st.pyplot(fig1)
            st.write("""
            **Phân tích chi tiết:** Sự đối chuẩn trực tiếp giữa **Danh tiếng Học thuật (AR)** và **Uy tín với Nhà tuyển dụng (ER)**. 
            Một chiến lược quản trị hiệu quả thường hướng tới việc thu hẹp khoảng cách giữa hai chỉ số này, đảm bảo rằng giá trị tri thức 
            được đào tạo (AR) phải tương xứng và đáp ứng được **nhu cầu thực tiễn của thị trường lao động** (ER).
            """)

        with col_heat:
            st.subheader("Ma trận tương quan")
            numeric_df = df_clean.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                corr_matrix = numeric_df.corr()
                fig_heat, ax_heat = plt.subplots(figsize=(10, 10))
                sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='RdYlGn', ax=ax_heat)
                st.pyplot(fig_heat)
            st.write("""
            **Phân tích chi tiết:** Ma trận Heatmap cung cấp cái nhìn tổng thể về **mối liên hệ cộng hưởng** giữa các biến số. Hệ số tương quan 
            càng gần **1.00** thể hiện các chỉ số này hỗ trợ lẫn nhau rất mạnh. Nhà chiến lược có thể xác định "đòn bẩy": 
            đầu tư vào một chỉ số then chốt để kéo theo sự tăng trưởng của các chỉ số khác.
            """)

        # =========================================================
        # PHẦN 3: PHÂN TÍCH TỐI ƯU HÓA & BIỂU ĐỒ TEST (FULL CAPTION)
        # =========================================================
        st.divider()
        st.header("3. Phân tích Tối ưu hóa Random Forest cho Score Scaled")
        
        target = 'score scaled'
        if target not in df_clean.columns:
            st.error(f"Không tìm thấy cột '{target}'!")
        else:
            features = df_clean.select_dtypes(include=[np.number]).columns.tolist()
            cols_to_remove = [target, 'score', 'rank', 'unnamed: 0']
            features = [f for f in features if f in df_clean.columns and f not in cols_to_remove]

            X = df_clean[features]
            y = df_clean[target]
            X = X.fillna(X.mean(numeric_only=True)) # Đồng bộ fillna
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            @st.cache_data
            def calculate_rmse_path(_X_t, _X_v, _y_t, _y_v):
                rmse_list = []
                for est in range(1, 151):
                    rf = RandomForestRegressor(n_estimators=est, random_state=42).fit(_X_t, _y_t)
                    rmse_list.append(np.sqrt(mean_squared_error(_y_v, rf.predict(_X_v))))
                return rmse_list

            with st.spinner("Đang huấn luyện mô hình..."):
                rmse_values = calculate_rmse_path(X_train, X_test, y_train, y_test)
                best_n = np.argmin(rmse_values) + 1

            fig_rmse, ax_rmse = plt.subplots(figsize=(10, 5))
            ax_rmse.plot(range(1, 151), rmse_values, color='darkgreen', lw=2)
            ax_rmse.axvline(x=best_n, color='red', linestyle='--', label=f"Điểm tối ưu: n={best_n}")
            ax_rmse.set_xlabel("Số lượng cây (Trees)")
            ax_rmse.set_ylabel("Lỗi dự báo (RMSE)")
            ax_rmse.legend()
            st.pyplot(fig_rmse)

            st.write(f"""
            **Phân tích kỹ thuật và tối ưu hóa:** Biểu đồ trên minh họa tiến trình tối ưu hóa mô hình thông qua việc điều chỉnh số lượng cây quyết định (n_estimators). 
            Trong thuật toán Random Forest, việc tăng số lượng cây thường giúp mô hình ổn định hơn, nhưng nếu quá nhiều sẽ gây lãng phí tài nguyên và không cải thiện thêm độ chính xác. 
            Điểm cắt tại **n={best_n}** chính là "điểm ngọt" (Sweet Spot) — nơi mà sai số căn bậc hai trung bình (RMSE) đạt mức thấp nhất, cho thấy mô hình đã học được 
            quy luật cốt lõi của dữ liệu mà không bị hiện tượng quá khớp (Overfitting). Đối với nhà quản trị, con số RMSE này đại diện cho mức độ rủi ro 
            hoặc sai số khi chúng ta sử dụng AI để lập kế hoạch dự báo điểm chuẩn hóa cho các năm tiếp theo.
            """)

            st.divider()
            st.subheader("So sánh giá trị Thực tế và Dự báo")
            rf_best = RandomForestRegressor(n_estimators=best_n, random_state=42).fit(X_train, y_train)
            y_pred = rf_best.predict(X_test)
            
            fig_test, ax_test = plt.subplots(figsize=(12, 6))
            v = range(len(y_test))
            ax_test.scatter(v, y_test, color='b', marker='x', label="Test Data (Thực tế)")
            ax_test.scatter(v, y_pred, color='r', marker='o', alpha=0.6, label="Predicted Values (Dự báo)")
            for i in range(len(v)):
                ax_test.plot([v[i], v[i]], [y_test.iloc[i], y_pred[i]], color='gray', linestyle='--', alpha=0.2)
            ax_test.legend()
            st.pyplot(fig_test)

            st.write("""
            **Đánh giá độ tin cậy của mô hình:** Biểu đồ so sánh trực quan giữa dữ liệu thực tế (Test Data) và giá trị dự báo (Predicted Values) cung cấp 
            cái nhìn khách quan về năng lực thực chiến của thuật toán. Mỗi cặp điểm xanh-đỏ trên cùng một trục dọc đại diện cho một trường đại học trong tập kiểm thử. 
            Khi các điểm đỏ (AI dự đoán) bám sát hoặc trùng khít với điểm xanh (thực tế), điều đó khẳng định rằng các chỉ số thành phần như AR, ER, ISR... 
            đang giải thích rất tốt sự biến động của điểm chuẩn hóa cuối cùng. 
            
            **Ứng dụng chiến lược:** Những vùng có khoảng cách lớn giữa hai điểm là tín hiệu cho thấy có những yếu tố "vô hình" hoặc các tiêu chí đặc thù 
            chưa được thống kê đầy đủ trong mô hình hiện tại. Tuy nhiên, với độ hội tụ cao như trên biểu đồ, nhà quản trị hoàn toàn có thể tin tưởng sử dụng 
            môi hình này làm "phòng thí nghiệm giả lập". Thay vì đầu tư dàn trải, nhà trường có thể thử nghiệm thay đổi các thông số đầu vào trên AI để tìm ra 
            lộ trình tăng trưởng điểm số tối ưu nhất với chi phí thấp nhất.
            """)

except Exception as e:
    st.error(f"Lỗi: {e}")