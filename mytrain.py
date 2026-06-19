from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r"D:\yolo26\yolo26n.pt")
    model.train(
        data=r"D:\yolo26\hmbb.yaml",
        epochs=500,
        imgsz=640,
        batch=-1,
        cache=False,#或"ram"
        workers=1,
       # --- 策略参数 ---
        patience=0,  # 早停机制
        # --- 数据增强 ---
        degrees=10.0,  # 旋转
        mosaic=1.0,  # 马赛克增强
        close_mosaic=25,  # 最后20轮关闭马赛克
        scale=0.5,
        device=0,  # 使用第一张显卡
        plots=True,  # 绘制训练曲线

        weight_decay=0.0005,# 正则化
        augment=True,

    )

