import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("GPU可用: ", tf.test.is_gpu_available())
print("可用GPU: ", tf.config.list_physical_devices('GPU'))
# 配置参数
IMAGE_SIZE = (32, 32)
BATCH_SIZE = 128
EPOCHS =30
NUM_CLASSES = 10
MODEL_SAVE_PATH = 'models/cifar10_resnet_model.keras'
CLASS_NAMES = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']

# 创建模型保存目录
os.makedirs('models', exist_ok=True)

# 加载CIFAR-10数据集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# 数据预处理
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 标签转为one-hot编码
y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

# 数据增强
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)
datagen.fit(x_train)

# 残差块
def residual_block(x, filters, stride=1, downsample=None):
    residual = x
    
    x = layers.Conv2D(filters, (3, 3), strides=stride, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(filters, (3, 3), strides=1, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    
    if downsample:
        residual = downsample(residual)
    
    x = layers.Add()([x, residual])
    x = layers.Activation('relu')(x)
    return x

# 构建ResNet模型
def build_resnet(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    
    # 初始卷积层
    x = layers.Conv2D(64, (3, 3), strides=1, padding='same', kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # 残差块
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    
    # 下采样并增加滤波器数量
    downsample = models.Sequential([
        layers.Conv2D(128, (1, 1), strides=2, padding='same', kernel_initializer='he_normal'),
        layers.BatchNormalization()
    ])
    x = residual_block(x, 128, stride=2, downsample=downsample)
    x = residual_block(x, 128)
    
    # 下采样并增加滤波器数量
    downsample = models.Sequential([
        layers.Conv2D(256, (1, 1), strides=2, padding='same', kernel_initializer='he_normal'),
        layers.BatchNormalization()
    ])
    x = residual_block(x, 256, stride=2, downsample=downsample)
    x = residual_block(x, 256)
    
    # 全局平均池化和输出层
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu', kernel_initializer='he_normal')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

# 创建模型
model = build_resnet((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), NUM_CLASSES)

# 编译模型
model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 回调函数
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

model_checkpoint = ModelCheckpoint(
    MODEL_SAVE_PATH,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-6
)

# 训练模型
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=x_train.shape[0] // BATCH_SIZE,
    validation_data=(x_test, y_test),
    epochs=EPOCHS,
    callbacks=[early_stopping, model_checkpoint, reduce_lr]
)

# 评估模型
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"测试集准确率: {test_acc:.4f}")

# 绘制训练曲线
def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    # 准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='训练准确率')
    plt.plot(history.history['val_accuracy'], label='验证准确率')
    plt.title('模型准确率')
    plt.xlabel('Epoch')
    plt.ylabel('准确率')
    plt.legend()
    
    # 损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title('模型损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

# 绘制训练曲线
plot_training_history(history)
