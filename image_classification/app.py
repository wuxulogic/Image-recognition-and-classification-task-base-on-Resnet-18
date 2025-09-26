from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# 配置
MODEL_PATH = 'models/cifar10_resnet_model.keras'
UPLOAD_FOLDER = 'uploads'
IMAGE_SIZE = (32, 32)
CLASS_NAMES = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']

# 创建上传文件夹
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 加载模型
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("模型加载成功")
except Exception as e:
    print(f"模型加载失败: {e}")
    model = None

# 图像预处理
def preprocess_image(image):
    # 调整大小
    image = image.resize(IMAGE_SIZE)
    # 转换为数组
    image_array = np.array(image)
    # 归一化
    image_array = image_array.astype('float32') / 255.0
    # 扩展维度以匹配模型输入
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': '模型加载失败，请检查模型文件'})
    
    if 'image' not in request.files:
        return jsonify({'error': '未找到图像文件'})
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': '未选择图像文件'})
    
    try:
        # 读取图像
        image = Image.open(io.BytesIO(file.read()))
        # 确保图像是RGB格式
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 预处理图像
        processed_image = preprocess_image(image)
        
        # 预测
        predictions = model.predict(processed_image)
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_index])
        
        # 准备所有类别的预测结果
        all_predictions = [(CLASS_NAMES[i], float(predictions[0][i])) for i in range(len(CLASS_NAMES))]
        
        # 返回结果
        return jsonify({
            'class': CLASS_NAMES[predicted_class_index],
            'confidence': confidence,
            'predictions': all_predictions
        })
    
    except Exception as e:
        return jsonify({'error': f'处理图像时出错: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)
