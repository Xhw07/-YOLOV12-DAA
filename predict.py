# -*- coding: utf-8 -*-
"""
YOLO 单张图片完整推理代码
功能：加载模型 -> 预测图片 -> 提取所有检测目标的置信度 -> 输出结果
适用：Ultralytics YOLOv8 (官方/自定义训练模型通用)
"""

# 1. 导入依赖库
from ultralytics import YOLO
import os

# ===================== 【核心配置项】修改这里即可 =====================
# 模型路径（官方模型：yolov8n.pt/yolov8s.pt；自定义训练模型填你的best.pt）
MODEL_PATH = "E:/Users/hewei/PycharmProjects/pythonProject2/YOLOV12/runs/detect/yolov12_attention10/weights/best.pt"
# 待推理的图片路径（替换成你的图片地址）
IMAGE_PATH = "E:/Users/hewei/PycharmProjects/pythonProject2/YOLOV12/datasets/valid/images/002104_jpg.rf.35f38e8921bc3c2bf649a515b380f034.jpg"
# 置信度阈值（低于该值的结果会被过滤，默认0.25）
CONF_THRESHOLD = 0.25
# ====================================================================

def run_inference():
    """
    完整推理主函数
    返回：检测结果列表（包含类别、置信度、坐标）
    """
    # 2. 异常检查：判断文件是否存在
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ 错误：图片不存在！路径：{IMAGE_PATH}")
        return None
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误：模型文件不存在！路径：{MODEL_PATH}")
        return None

    try:
        # 3. 加载训练好的模型（推理第一步）
        print("🔄 正在加载模型...")
        model = YOLO(MODEL_PATH)

        # 4. 执行核心推理（模型预测图片，核心代码）
        print(f"🔄 正在对图片 {IMAGE_PATH} 进行推理...")
        results = model(
            source=IMAGE_PATH,
            conf=CONF_THRESHOLD,
            verbose=False  # 关闭模型冗余日志
        )

        # 5. 解析推理结果，提取置信度
        print("\n==================== 推理结果 ====================")
        detection_results = []
        # 获取单张图片的推理结果
        result = results[0]

        # 遍历所有检测框
        for box in result.boxes:
            # 提取核心数据
            class_id = int(box.cls[0])          # 类别ID
            confidence = box.conf[0].item()      # 置信度（0~1，核心需求）
            class_name = model.names[class_id]   # 类别名称
            x1, y1, x2, y2 = box.xyxy[0].tolist()# 坐标位置

            # 保存结果
            detection_info = {
                "类别": class_name,
                "置信度": round(confidence, 4),
                "坐标": (round(x1), round(y1), round(x2), round(y2))
            }
            detection_results.append(detection_info)

            # 打印结果
            print(f"✅ 类别：{class_name} | 置信度：{confidence:.4f} | 坐标：{detection_info['坐标']}")

        if not detection_results:
            print("ℹ️ 未检测到任何目标")
        print("==================================================\n")

        return detection_results

    except Exception as e:
        print(f"❌ 推理失败，错误信息：{str(e)}")
        return None

# 6. 运行推理
if __name__ == "__main__":
    result = run_inference()
    # 可直接使用result中的置信度数据