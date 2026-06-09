import torch
import warnings
warnings.filterwarnings("ignore")
from ultralytics import YOLO
from ultralytics.nn.modules import C3k2
# -------------------------- 1. 导入注意力模块（请替换为你的实际路径）--------------------------
# 示例：假设你的注意力模块在 attention_modules.py 中
from attention_modules import DecoupledCSAttention, AgentAttention  # 按需修改


# -------------------------- 2. 核心配置参数（必须按你的任务修改）--------------------------
class Config:
    # 模型配置
    base_model = "yolov12n.pt"  # 基础模型权重（yolov12n/s/m/l/x.pt）
    custom_yaml = "custom_yolov12.yaml"  # 对齐后的yaml配置文件路径
    attention_paths = [  # 实际检测到的3个C3k2层路径（从日志复制）
        "model.2",  # 正确格式：C3k2（小写k）
        "model.4",
        "model.20"
    ]
    attention_types = [  # 对应每个C3k2层的注意力模块类型（和日志一致）
        DecoupledCSAttention,
        AgentAttention,
        DecoupledCSAttention
    ]

    # 训练配置
    data_yaml = "E:/Users/hewei/PycharmProjects/pythonProject2/YOLOV12/dataset.yaml"  # 数据集配置
    epochs = 100  # 训练轮次
    batch = 8 # 批次大小（根据GPU显存调整，显存不够改小）
    imgsz = 320  # 输入图像尺寸
    device = 0  # 训练设备（0=第1块GPU，-1=CPU，多GPU写[0,1]）
    lr0 = 0.001# 初始学习率
    lrf = 0.01
    weight_decay = 0.0005  # 权重衰减（防止过拟合）
    patience = 50  # 早停耐心值（50轮没提升就停止）
    amp = True
    save = True  # 是否保存权重
    works = 2
    mosaic = 0.0
    plot = False
    project = "runs/detect"  # 训练结果保存路径
    name = "yolov12_attention"  # 实验名称
    exist_ok = False  # 是否覆盖已有实验文件夹


# -------------------------- 3. 辅助函数：通过路径获取模型模块 --------------------------
def get_module_by_path(model, path: str):
    """简洁兼容版：解析YOLOv12真实路径（如model.2）"""
    path_parts = path.split(".")
    current_module = model  # 初始化变量，避免未赋值报错

    for part in path_parts:
        try:
            if part.isdigit():
                current_module = current_module[int(part)]  # 处理索引（如2）
            else:
                current_module = getattr(current_module, part)  # 处理属性（如model）
        except (IndexError, AttributeError, TypeError) as e:
            raise ValueError(f"路径 {path} 解析失败：{str(e)}") from e

    return current_module


# -------------------------- 4. 核心函数：插入注意力模块并融入前向传播 --------------------------
# 修正后的 insert_attention_to_c3k2 函数（核心部分）
def insert_attention_to_c3k2(model, config):
    from ultralytics.nn.modules import C3k2
    c3k2_out_channels = {
        "model.2": 64,
        "model.4": 128,
        "model.20": 256
    }
    print("\n🔧 开始插入注意力模块...")
    for idx, (path, attn_type) in enumerate(zip(config.attention_paths, config.attention_types)):
        try:
            target_c3k2 = get_module_by_path(model, path)
            if not isinstance(target_c3k2, C3k2):
                print(f"⚠️  路径{path}不是C3k2层，跳过")
                continue

            out_channels = c3k2_out_channels[path]
            # 关键修改：将 in_channels 改为 dim（或模块实际的参数名）
            target_c3k2.attention = attn_type(dim=out_channels)  # 替换 in_channels → dim
            print(f"✅ 路径={path} → 插入{attn_type.__name__}（输入维度：{out_channels}）")

            # 重写forward方法（不变）
            def new_forward(self, x):
                y1 = self.cv1(x)
                y2 = self.cv2(x).chunk(2, 1)[0]
                for m in self.m:
                    y2 = m(y2)
                out = torch.cat((y1, y2), 1)
                out = self.attention(out)
                return out

            target_c3k2.forward = new_forward.__get__(target_c3k2)

        except Exception as e:
            print(f"❌ 路径{path}插入注意力失败：{str(e)}")

    verify_attention_insertion(model)
    return model


# 辅助函数：验证注意力模块是否插入成功（保持不变）
def verify_attention_insertion(model):
    """验证C3k2层是否成功插入注意力模块"""
    for name, module in model.named_modules():
        if isinstance(module, C3k2):
            has_attention = hasattr(module, "attention") and "Attention" in str(module.attention.__class__.__name__)
            status = "✅ 已插入" if has_attention else "❌ 未插入"
            attn_name = module.attention.__class__.__name__ if has_attention else "无"
            print(f"  - C3k2层路径：{name} → 注意力模块：{attn_name} {status}")


# -------------------------- 5. 辅助函数：验证注意力模块是否插入成功 --------------------------


# -------------------------- 6. 主训练流程 --------------------------
# -------------------------- 6. 主训练流程（修正版） --------------------------
if __name__ == "__main__":
    # 加载配置
    cfg = Config()

    # 1. 加载基础模型（指定任务为detect，避免None报错）
    print(f"✅ 加载基础模型：{cfg.base_model}")
    yolo_model = YOLO(cfg.base_model, task="detect")  # 关键：指定task="detect"
    model = yolo_model.model  # 获取核心检测网络（nn.Module）


    # 2. 打印C3k2真实路径（验证路径正确性）
    def print_c3k2_real_paths(model):
        from ultralytics.nn.modules import C3k2
        print("\n📌 模型中C3k2层的真实路径：")
        c3k2_real_paths = []
        for name, module in model.named_modules():
            if isinstance(module, C3k2):
                print(f"  - {name}")
                c3k2_real_paths.append(name)
        print(f"\n📋 推荐attention_paths：{c3k2_real_paths}")


    print_c3k2_real_paths(model)

    # 3. 插入注意力模块（用真实路径，确保Config中的attention_paths已更新）
    print("\n🔧 开始插入注意力模块...")
    try:
        model = insert_attention_to_c3k2(model, cfg)
        yolo_model.model = model  # 替换为插入注意力后的模型
        print("✅ 注意力模块插入成功！")
    except Exception as e:
        print(f"❌ 注意力插入失败：{str(e)}")
        exit(1)

    # 4. 关联自定义yaml配置（无需重新加载，直接替换cfg）
    print(f"\n✅ 加载对齐后的配置文件：{cfg.custom_yaml}")
    yolo_model.cfg = cfg.custom_yaml  # 关联自定义配置
    yolo_model.task = "detect"  # 再次确认任务类型

    # 5. 开始训练（参数不变）
    print(f"\n🚀 开始训练（设备：GPU {cfg.device}）")
    results = yolo_model.train(
        data=cfg.data_yaml,
        epochs=cfg.epochs,
        batch=cfg.batch,
        imgsz=cfg.imgsz,
        device=cfg.device,
        lr0=cfg.lr0,
        lrf=cfg.lrf,
        weight_decay=cfg.weight_decay,
        patience=cfg.patience,
        save=cfg.save,
        project=cfg.project,
        name=cfg.name,
        exist_ok=cfg.exist_ok,
        verbose=True
    )

    # 6. 训练完成验证
    print("\n🎉 训练完成！开始验证...")
    val_results = yolo_model.val()
    print(f"验证集mAP50：{val_results.box.map50:.3f} | mAP50-95：{val_results.box.map:.3f}")
# -------------------------- 7. 注意力模块示例（如果没有实现，新建 attention_modules.py）--------------------------
"""
# attention_modules.py（放在与训练代码同目录）
import torch
import torch.nn as nn

class DecoupledCSAttention(nn.Module):
    '''轻量解耦通道-空间注意力模块'''
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_attn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
        # 空间注意力
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, 3, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        ca = self.avg_pool(x)
        ca = self.channel_attn(ca)
        x_ca = x * ca

        # 空间注意力
        sa = torch.cat([torch.mean(x, dim=1, keepdim=True), torch.max(x, dim=1, keepdim=True)[0]], dim=1)
        sa = self.spatial_attn(sa)
        x_sa = x * sa

        return x_ca + x_sa  # 融合输出

class AgentAttention(nn.Module):
    '''轻量代理注意力模块'''
    def __init__(self, in_channels, num_agents=8):
        super().__init__()
        self.num_agents = num_agents
        self.agent_conv = nn.Conv2d(in_channels, num_agents, 1, bias=False)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.output_conv = nn.Conv2d(in_channels, in_channels, 1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        # 生成代理特征
        agents = self.agent_conv(x).view(B, self.num_agents, -1)  # (B, A, H*W)
        agents = torch.softmax(agents, dim=-1).unsqueeze(1)  # (B, 1, A, H*W)

        # 价值特征
        value = self.value_conv(x).view(B, C, -1).unsqueeze(-2)  # (B, C, 1, H*W)

        # 注意力聚合
        attn = torch.matmul(value, agents.transpose(-1, -2))  # (B, C, 1, A)
        attn = torch.mean(attn, dim=-1, keepdim=True)  # (B, C, 1, 1)
        attn = torch.sigmoid(attn)

        return self.output_conv(x * attn)
"""