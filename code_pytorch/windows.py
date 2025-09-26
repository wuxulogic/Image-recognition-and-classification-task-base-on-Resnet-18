from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
import os
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# 定义不同版本的模型文件路径
MODEL_PATHS = {
    '1': 'models/cifar10_resnet_pytorch_optimized_v1.pth',
    '2': 'models/cifar10_resnet_pytorch_optimized_v2.pth'
}

# --- 运行时选择模型 ---
selected_version = input("请输入要加载的模型版本 (1 或 2): ")

if selected_version not in MODEL_PATHS:
    print("输入无效，将默认加载版本 1。")
    selected_version = '1'

MODEL_PATH = MODEL_PATHS[selected_version]
print(f"正在加载模型: {MODEL_PATH}")
# 获取当前选定的模型路径
UPLOAD_FOLDER = 'uploads'
IMAGE_SIZE = (32, 32)
CLASS_NAMES = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']

# 创建上传文件夹
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 定义 PyTorch 模型架构 (这部分必须和训练时完全一致)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

# 加载 PyTorch 模型
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet18().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("模型加载成功")
except Exception as e:
    print(f"模型加载失败: {e}")
    model = None


# 图像预处理 (为 PyTorch 重新编写)
def preprocess_image(image):
    # PyTorch 训练时的归一化参数
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    # 应用变换，并添加批次维度
    tensor_image = transform(image).unsqueeze(0)
    return tensor_image

@app.route('/')
def index():
    return render_template('index_v2.html')


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
        processed_image = preprocess_image(image).to(device)

        # 预测
        with torch.no_grad():
            outputs = model(processed_image)
            predictions = torch.softmax(outputs, dim=1)

        predicted_class_index = torch.argmax(predictions, dim=1).item()
        confidence = predictions[0][predicted_class_index].item()

        # 准备所有类别的预测结果
        all_predictions = [(CLASS_NAMES[i], predictions[0][i].item()) for i in range(len(CLASS_NAMES))]

        # 返回结果
        return jsonify({
            'class': CLASS_NAMES[predicted_class_index],
            'confidence': confidence,
            'predictions': all_predictions
        })

    except Exception as e:
        return jsonify({'error': f'处理图像时出错: {str(e)}'})


if __name__ == '__main__':
    app.run(debug=False)