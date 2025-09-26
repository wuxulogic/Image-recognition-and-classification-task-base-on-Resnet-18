# Image-recognition-and-classification-task-base-on-Resnet-18
本项目在原始ResNet18的基础上，通过数据增强策略、小尺寸图像适配、双框架兼容、全栈部署等多维度的修改与扩展，不仅提升了模型在CIFAR-10任务上的精度与泛化能力，更将其从策略模型升级为客直接应用的图象识别分类系统。  

基于ResNet18残差网络实现CIFAR-10数据集分类，pytorch版本测试准确率达到93.37%，tensorflow版本测试准确率达到91%（预计），在CIFAR-10数据集上属于良好水平，验证了残差链接在小尺寸图象特征提取中的有效性。  
构建了“前端交互-后端推理-模型支撑”全栈结构，前端支持图像上传，后端通过Flask提供高效推理接口。  
同时完成了pytorch和tensorflow版本的模型开发，可以为不同的技术栈团队提供可复用的代码模板。  
实现了结果可视化，响应式界面等工程特性，为开发者和用户提供了友好的参考与交互界面。 

但是，由于设备等方面的局限性，模型仍存在一些不足  
模型目前仅支持CIFAR-10定义的10类物体识别，对于高分辨率图象，复杂遮挡及光照变化等场景鲁棒性不足，真实操作与测试环境受限于训练数据的规模与多样性。  
系统工程化程度有待提升。  
pytorch模型加载依赖完整代码环境，受设备性能影响，tensorflow版本在CPU与GPU下均未实现完整的训练。  
模型对大尺寸图象的适配性较差，需要额外修改特征提取层结构。  

---
下载模型后可直接全局环境/虚拟环境（可选）训练运行  
一、首先训练模型	<br>	**python pytorch.py**  <br>自动下载 CIFAR-10 数据集到./data目录  
训练 50 个 epochs，使用 ResNet18 模型  
自动保存验证集准确率最高的模型到models/cifar10_resnet_pytorch_optimized.pth  
训练完成后生成训练曲线图片training_history_pytorch_optimized.png  
支持 GPU 加速（自动检测，优先使用 GPU）  
训练后结果如下
<img width="1200" height="400" alt="曲线" src="https://github.com/user-attachments/assets/cbe240c3-149d-4587-81cd-6caeddd12bff" />
<img width="729" height="511" alt="准确率" src="https://github.com/user-attachments/assets/95f975aa-54aa-4189-abe3-bd691f816008" />
  
二、启动web服务  
**python windows.py**  
运行后会提示输入模型版本（1 或 2）  
输入无效时默认加载版本 1  
模型加载成功后，服务启动在默认端口（5000）  
<img width="1734" height="927" alt="屏幕截图 2025-09-26 211631" src="https://github.com/user-attachments/assets/b84cacfa-7d14-40f3-ba71-d0203f84e347" />
随后在浏览器打开网页即可上传图象识别  
以下是部分识别结果<img width="2560" height="1528" alt="屏幕截图 2025-09-25 185448" src="https://github.com/user-attachments/assets/fd1bc36c-c4cf-4cd6-97f8-59186165a8b8" />
<img width="2560" height="1528" alt="屏幕截图 2025-09-25 185426" src="https://github.com/user-attachments/assets/4b94702d-c7e8-4f8e-a021-0eb9cd6f5d3f" />
<img width="2560" height="1528" alt="屏幕截图 2025-09-25 185413" src="https://github.com/user-attachments/assets/961aecbb-2155-4945-90a9-691a1b9a7b52" />
<img width="2560" height="1528" alt="屏幕截图 2025-09-25 185353" src="https://github.com/user-attachments/assets/b7f56a15-0bd4-476f-b1b2-b91f174d9fdb" />
<img width="2560" height="1528" alt="屏幕截图 2025-09-25 185319" src="https://github.com/user-attachments/assets/01c4ae9e-b1cc-4058-ad88-e7636ded2a01" />
