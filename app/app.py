import streamlit as st
import pandas as pd
import joblib

# 1. CẤU HÌNH
st.set_page_config(page_title="Dự đoán Bỏ học", page_icon="🎓")
st.title("🎓 Dự báo Nguy cơ Bỏ học")
st.write("Nhập 10 thông tin quan trọng nhất để kiểm tra nhanh.")

# 2. LOAD PIPELINE RÚT GỌN
@st.cache_resource
def load_pipeline():
    try:
        return joblib.load('dropout_pipeline_lite.pkl')
    except:
        st.error("⚠️ Chưa thấy file 'dropout_pipeline_lite.pkl'. Hãy chạy file train_lite.py trước!")
        return None

pipeline = load_pipeline()

# 3. FORM NHẬP LIỆU (CHIA 2 CỘT CHO ĐẸP)
col1, col2 = st.columns(2)

with col1:
    st.subheader("📚 Kết quả học tập")
    # Học kỳ 1
    sem1_appr = st.number_input("Số môn đậu Kỳ 1", min_value=0, value=5, help="Số môn đã qua trong kỳ 1")
    sem1_grade = st.number_input("Điểm trung bình Kỳ 1", min_value=0.0, max_value=20.0, value=12.0)
    
    # Học kỳ 2 (Quan trọng nhất)
    sem2_appr = st.number_input("Số môn đậu Kỳ 2", min_value=0, value=5, help="Số môn đã qua trong kỳ 2")
    sem2_grade = st.number_input("Điểm trung bình Kỳ 2", min_value=0.0, max_value=20.0, value=12.0)
    
    st.caption("*Lưu ý: Nếu chưa có điểm kỳ 2, hãy nhập dự kiến hoặc lấy bằng kỳ 1.")

with col2:
    st.subheader("💰 Tài chính & Cá nhân")
    tuition = st.selectbox("Đóng học phí đầy đủ?", [1, 0], format_func=lambda x: "✅ Đã đóng" if x==1 else "❌ Còn nợ")
    debtor = st.selectbox("Có nợ môn/nợ tiền không?", [0, 1], format_func=lambda x: "Không nợ" if x==0 else "Có nợ")
    scholarship = st.selectbox("Có học bổng không?", [0, 1], format_func=lambda x: "Không" if x==0 else "Có học bổng")
    
    st.divider()
    age = st.number_input("Tuổi nhập học", min_value=17, value=18)
    gender = st.radio("Giới tính", [1, 0], format_func=lambda x: "Nam" if x==1 else "Nữ", horizontal=True)
    displaced = st.checkbox("Sinh viên sống xa nhà?", value=True)
    displaced_val = 1 if displaced else 0

# 4. DỰ ĐOÁN
if st.button("🚀 PHÂN TÍCH NGAY", type="primary", use_container_width=True):
    if pipeline:
        # Tạo DataFrame đúng 10 cột như lúc train
        input_data = pd.DataFrame({
            "Tuition fees up to date": [tuition],
            "Curricular units 2nd sem (approved)": [sem2_appr],
            "Curricular units 2nd sem (grade)": [sem2_grade],
            "Curricular units 1st sem (approved)": [sem1_appr],
            "Curricular units 1st sem (grade)": [sem1_grade],
            "Age at enrollment": [age],
            "Debtor": [debtor],
            "Scholarship holder": [scholarship],
            "Gender": [gender],
            "Displaced": [displaced_val]
        })

        try:
            # Dự đoán
            proba = pipeline.predict_proba(input_data)[0]
            risk_percent = proba[1] * 100
            
            # Hiển thị kết quả đẹp
            st.markdown("---")
            if risk_percent > 50:
                st.error(f"### ⚠️ CẢNH BÁO: NGUY CƠ BỎ HỌC CAO")
                st.write(f"Hệ thống đánh giá tỷ lệ rủi ro là: **{risk_percent:.1f}%**")
                st.progress(int(risk_percent), text="Mức độ nguy hiểm")
                st.info("💡 **Gợi ý:** Sinh viên này cần được cố vấn học tập hỗ trợ ngay lập tức về vấn đề tài chính hoặc cải thiện điểm số.")
            else:
                st.success(f"### ✅ AN TOÀN: TÌNH TRẠNG TỐT")
                st.write(f"Khả năng tiếp tục học/tốt nghiệp là: **{100 - risk_percent:.1f}%**")
                st.progress(int(100 - risk_percent), text="Độ an toàn")
                
        except Exception as e:
            st.error(f"Lỗi: {e}")