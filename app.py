from collections import Counter
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

st.set_page_config(page_title="Plant Disease Classification", page_icon="🌿")
st.title("🌿 ระบบจำแนกโรคพืช")
st.write(
    "อัปโหลดรูปภาพใบพืชเพื่อวิเคราะห์หาชนิดของโรค ด้วยระบบประมวลผลร่วมหลายโมเดล"
    " (Ensemble Model)"
)


# 1. โหลดโมเดลทั้ง 3 ตัวเข้าสู่ความจำระบบ
@st.cache_resource
def load_all_models():
  m_mobile = tf.keras.models.load_model("mobilenetv2_plant_disease.keras")
  m_eff = tf.keras.models.load_model("efficientnetb0_plant_disease.keras")
  m_cnn = tf.keras.models.load_model("cnn_baseline.keras")
  return m_mobile, m_eff, m_cnn


try:
  model_mobile, model_eff, model_cnn = load_all_models()
except Exception as e:
  st.error(f"ไม่สามารถโหลดไฟล์โมเดลได้: {e}")

CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Background_without_leaves",
    "Blueberry___healthy",
    "Cherry___healthy",
    "Cherry___Powdery_mildew",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn___Common_rust",
    "Corn___healthy",
    "Corn___Northern_Leaf_Blight",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___healthy",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___healthy",
    "Potato___Late_blight",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___healthy",
    "Strawberry___Leaf_scorch",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___healthy",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
]


# ฟังก์ชันทำนายผลแยกแต่ละโมเดลพร้อม Preprocessing ที่ถูกต้อง
def predict_single_model(model, image_array, model_type):
  # คัดลอก array แยกเพื่อไม่ให้การ Preprocess ของแต่ละโมเดลตีกัน
  img_input = image_array.copy()

  if model_type == "mobilenet":
    img_input = tf.keras.applications.mobilenet_v2.preprocess_input(img_input)
  elif model_type == "efficientnet":
    img_input = tf.keras.applications.efficientnet.preprocess_input(img_input)
  else:
    img_input = img_input / 255.0

  preds = model.predict(img_input, verbose=0)[0]
  idx = np.argmax(preds)
  conf = np.max(preds) * 100
  class_name = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"Unknown___{idx}"
  return class_name, conf


uploaded_file = st.file_uploader(
    "เลือกไฟล์รูปภาพใบพืช (jpg, jpeg, png)...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
  image = Image.open(uploaded_file)
  st.image(image, caption="รูปภาพที่อัปโหลด", use_container_width=True)

  if st.button("วิเคราะห์โรคพืช"):
    try:
      if image.mode != "RGB":
        image = image.convert("RGB")

      # ปรับขนาดภาพและแปลงเป็น NumPy Array
      resized_img = image.resize((128, 128))
      base_array = tf.keras.utils.img_to_array(resized_img)
      # ใช้ np.expand_dims เพื่อให้ยังคงเป็น NumPy Array
      base_array = np.expand_dims(base_array, axis=0)

      # ทำนายผลจากทั้ง 3 โมเดล
      c_mobile, conf_mobile = predict_single_model(
          model_mobile, base_array, "mobilenet"
      )
      c_eff, conf_eff = predict_single_model(
          model_eff, base_array, "efficientnet"
      )
      c_cnn, conf_cnn = predict_single_model(model_cnn, base_array, "cnn")

      predictions = [c_mobile, c_eff, c_cnn]
      confidences = [conf_mobile, conf_eff, conf_cnn]

      # นับจำนวนคำตอบจากทั้ง 3 โมเดล
      counts = Counter(predictions)
      most_common = counts.most_common()

      # กรณีที่ 1: มีเสียงส่วนใหญ่ (2 หรือ 3 โมเดลทายตรงกัน)
      if most_common[0][1] >= 2:
        chosen_class = most_common[0][0]
        # เลือกค่า confidence ที่สูงที่สุดเฉพาะจากโมเดลที่ทายตรงกับเสียงส่วนใหญ่
        matching_confs = [
            conf for p, conf in zip(predictions, confidences) if p == chosen_class
        ]
        chosen_conf = max(matching_confs)
      else:
        # กรณีที่ 2: ทั้ง 3 โมเดลทายไม่ตรงกันเลย ให้ยึด MobileNetV2 เป็นหลัก
        chosen_class = c_mobile
        chosen_conf = conf_mobile

      # จัดการข้อความผลลัพธ์
      parts = chosen_class.split("___")

      if chosen_class == "Background_without_leaves":
        plant_type = "ไม่พบใบพืช"
        disease = "ไม่ใช่ใบพืช (Background)"
        status = "ไม่สามารถวิเคราะห์ได้"
        advice = "กรุณาอัปโหลดรูปภาพที่มีใบพืชเพื่อทำการวิเคราะห์"
      else:
        plant_type = parts[0] if len(parts) > 1 else "ไม่ทราบชนิดพืช"
        disease = (
            parts[1].replace("_", " ") if len(parts) > 1 else chosen_class
        )

        if disease.lower() == "healthy":
          status = "พืชปกติ แข็งแรงดี"
          advice = "ดูแลรักษาตามปกติต่อไป"
          disease = "ปกติ (Healthy)"
        else:
          status = "มีแนวโน้มติดโรค"
          advice = (
              "ควรตรวจสอบบริเวณใบเพิ่มเติมและดำเนินการจัดการโรคตามแนวทางที่เหมาะสม"
          )

      st.success("วิเคราะห์เสร็จสิ้น")

      st.markdown("### ผลการวิเคราะห์ภาพ")
      st.write(f"**ชนิดพืช:** {plant_type.replace('_', ' ')}")
      st.write(f"**ผลการตรวจ:** {disease}")
      st.write(f"**ความมั่นใจ:** {chosen_conf:.2f}%")
      st.write(f"**สถานะ:** {status}")
      st.write(f"**คำแนะนำ:** {advice}")

    except Exception as e:
      st.error(f"เกิดข้อผิดพลาด: {e}")