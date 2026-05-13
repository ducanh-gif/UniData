import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# 1. CẤU HÌNH TRANG & MENU
st.set_page_config(page_title="Dự đoán Chiến lược", layout="wide")

def menu():
    st.sidebar.page_link("app.py", label=" Trang chủ")
    st.sidebar.page_link("pages/Data_Analysis.py", label=" Phân tích dữ liệu")
    st.sidebar.page_link("pages/Input_Record.py", label=" Thêm dữ liệu")
    st.sidebar.page_link("pages/Prediction.py", label=" Dự đoán")

menu()

st.title("Phòng Thí nghiệm Giả lập & Dự báo Chiến lược")
st.markdown("---")

# Dictionary chú thích các thuật ngữ
dict_help = {
    'ar score': 'Academic Reputation: Điểm uy tín học thuật dựa trên khảo sát chuyên gia.',
    'er score': 'Employer Reputation: Điểm uy tín đối với nhà tuyển dụng.',
    'isr score': 'International Student Ratio: Tỷ lệ sinh viên quốc tế (Quốc tế hóa).',
    'ifr score': 'International Faculty Ratio: Tỷ lệ giảng viên quốc tế.',
    'cpf score': 'Citations per Faculty: Số trích dẫn khoa học trên mỗi giảng viên.',
    'fsr score': 'Faculty Student Ratio: Tỷ lệ giảng viên trên sinh viên.',
    'ger score': 'Graduate Employment Outcomes: Tỷ lệ việc làm sau tốt nghiệp.',
    'score scaled': 'Tổng điểm cuối cùng sau khi đã được chuẩn hóa trên thang 100.'
}

# 2. LOAD DỮ LIỆU VÀ XỬ LÝ
try:
    df = pd.read_csv("UNI.csv")
    df.columns = [col.strip().lower() for col in df.columns]
    
    target_col = 'score scaled'
    
    if target_col not in df.columns:
        st.error(f"Không tìm thấy cột '{target_col}' trong file CSV!")
    else:
        df_clean = df.dropna(subset=[target_col]).copy()
        features = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        cols_to_exclude = [target_col, 'score', 'rank', 'unnamed: 0']
        features = [f for f in features if f in df_clean.columns and f not in cols_to_exclude]
        
        if not features:
            st.warning("Không tìm thấy các cột dữ liệu số để huấn luyện mô hình.")
        else:
            df_clean[features] = df_clean[features].fillna(df_clean[features].mean())

            # =========================================================
            # MỤC 1: LINEAR REGRESSION
            # =========================================================
            st.header(f"1. Mô phỏng Tuyến tính: AR vs {target_col.title()}")
            
            if 'ar score' in df_clean.columns:
                X_lin = df_clean[['ar score']].values
                y_lin = df_clean[target_col].values
                model_lin = LinearRegression().fit(X_lin, y_lin)

                col_l1, col_l2 = st.columns([1, 1.2])
                with col_l1:
                    st.write("**Giả lập mục tiêu:**")
                    target_ar = st.slider(
                        "Điều chỉnh mức AR Score (Học thuật):", 
                        0.0, 100.0, 80.0, 
                        help=dict_help.get('ar score')
                    )
                    pred_l = model_lin.predict([[target_ar]])[0]
                    st.metric(
                        f"Dự báo {target_col.upper()}", 
                        f"{max(0, pred_l):.2f}",
                        help=dict_help.get('score scaled')
                    )
                    
                    st.info("""
                    **Góc nhìn chiến lược:** Mô hình này giúp xác định "độ nhạy" của tổng điểm đối với uy tín học thuật. 
                    Nếu đường đỏ dốc lên mạnh, việc đầu tư vào các công trình nghiên cứu và bài báo khoa học sẽ mang lại 
                    hiệu quả thăng hạng nhanh nhất.
                    """)

                with col_l2:
                    fig_lin, ax_lin = plt.subplots(figsize=(8, 5))
                    sns.regplot(x='ar score', y=target_col, data=df_clean, 
                                scatter_kws={'alpha':0.3}, line_kws={'color':'red'}, ax=ax_lin)
                    ax_lin.scatter([target_ar], [pred_l], color='yellow', s=200, edgecolors='black', label="Điểm giả lập")
                    ax_lin.set_xlabel("AR Score (Học thuật)")
                    ax_lin.set_ylabel("Score Scaled (Tổng điểm)")
                    ax_lin.legend()
                    st.pyplot(fig_lin)
            
            st.divider()

            # =========================================================
            # MỤC 2: RANDOM FOREST
            # =========================================================
            st.header(f"2. Dự đoán Đa biến bằng Random Forest")
            
            X_rf = df_clean[features]
            y_rf = df_clean[target_col]
            model_rf_final = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_rf, y_rf)

            col_r1, col_r2 = st.columns([1, 1.2])
            
            with col_r1:
                st.subheader("Bàn điều khiển thông số")
                input_data = {}
                for col in features:
                    avg_val = float(df_clean[col].mean())
                    # Thêm chú thích vào tooltips (help) của từng ô nhập liệu
                    help_text = dict_help.get(col, f"Chỉ số {col.upper()} của trường.")
                    input_data[col] = st.number_input(
                        f"{col.upper()}:", 
                        0.0, 100.0, avg_val, 
                        key=f"input_{col}",
                        help=help_text
                    )
                
                if st.button("🚀 Chạy dự báo AI"):
                    input_df = pd.DataFrame([input_data])
                    pred_rf = model_rf_final.predict(input_df)[0]
                    st.divider()
                    st.success(f"Kết quả dự báo {target_col.upper()}: **{pred_rf:.2f}**")
                    
                    st.write(f"""
                    **Phân tích ứng dụng:** Dựa trên các chỉ số bạn nhập, AI dự báo trường sẽ đạt mức điểm **{pred_rf:.2f}**. 
                    Đây là kết quả của sự tổng hòa tất cả các yếu tố, cho thấy tính thực tế cao hơn so với việc chỉ nhìn vào một chỉ số đơn lẻ.
                    """)

            with col_r2:
                st.subheader("Ma trận Trọng số (Feature Importance)")
                importances = model_rf_final.feature_importances_
                feat_imp = pd.Series(importances, index=features).sort_values(ascending=True)
                
                fig_imp, ax_imp = plt.subplots(figsize=(8, 8))
                colors = plt.cm.Oranges(np.linspace(0.4, 0.8, len(feat_imp)))
                
                # Lấy tên tiếng Việt từ dictionary, nếu không có thì dùng tên gốc viết hoa
                display_name = dict_help.get(col, col.upper())
                # Chỉnh sửa tên hiển thị trên biểu đồ để có chú thích tiếng Việt
                friendly_labels = [f"{idx.upper()} ({dict_help.get(idx, '').split(':')[0]})" for idx in feat_imp.index]
                
                ax_imp.barh(friendly_labels, feat_imp.values, color=colors)
                ax_imp.set_title("Chỉ số nào ảnh hưởng lớn nhất đến Tổng điểm?")
                st.pyplot(fig_imp)
                
                st.write(f"""
                **Lời khuyên:** Chỉ số **{feat_imp.index[-1].upper()}** đang đóng vai trò quan trọng nhất trong việc hình thành 
                điểm số. Để tối ưu hóa lộ trình thăng hạng, nhà trường cần tập trung cải thiện chỉ số này trước tiên.
                """)

except Exception as e:
    st.error(f"Lỗi thực thi: {e}")