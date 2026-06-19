# ConDSeg + YOLOv11s 射线底片缺陷评片网站

这是一个把现有 Python 评片脚本接入网站的前后端工程。

## 网站怎么工作

```text
浏览器网页 Vue
  -> 上传射线底片
FastAPI 后端
  -> 调用 D:\yolo26\评定脚本最终版.py 的评片逻辑
  -> 加载训练好的 best.pt 模型
  -> 生成缺陷列表和标注图
网页
  -> 展示原图、标注图、缺陷明细
  -> 人工确认/改类别/删误检/写备注
  -> 导出 Excel 台账
```

模型和评片代码在后端运行，不会放进前端网页里。

## 目录

```text
backend/
  app/
    main.py                FastAPI 接口
    inference_adapter.py   评片脚本适配器
    excel_export.py        Excel 台账导出
  weights/
    best.pt                推荐把训练好的模型放这里
  runtime/
    jobs/                  上传图片、标注图、临时任务结果

frontend/
  src/
    App.vue                评片工作台页面
```

## 准备模型

推荐把训练好的模型复制到：

```text
backend\weights\best.pt
```

如果这里没有模型，后端会尝试使用你原来脚本旁边的：

```text
D:\yolo26\models1\best.pt
```

也可以通过环境变量指定：

```powershell
$env:MODEL_WEIGHTS_PATH="D:\yolo26\models1\best.pt"
```

评片脚本默认使用：

```text
D:\yolo26\评定脚本最终版.py
```

如果脚本换位置，可以设置：

```powershell
$env:RATING_SCRIPT_PATH="D:\yolo26\评定脚本最终版.py"
```

## 启动后端

进入后端目录：

```powershell
cd C:\Users\gugu\Documents\Codex\2026-06-07\condseg-yolov11s\backend
```

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

启动后端：

```powershell
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

后端启动后，接口地址是：

```text
http://localhost:8000
```

局域网其他电脑访问时，把 `localhost` 换成这台服务器的 IP。

## 启动前端

打开另一个 PowerShell，进入前端目录：

```powershell
cd C:\Users\gugu\Documents\Codex\2026-06-07\condseg-yolov11s\frontend
```

安装依赖：

```powershell
npm install
```

启动网页：

```powershell
npm run dev
```

浏览器打开 Vite 显示的地址，通常是：

```text
http://localhost:5173
```

## 使用流程

1. 打开网页。
2. 选择一张或多张射线底片。
3. 点击“开始评定”。
4. 查看原图、模型标注图和缺陷明细。
5. 人工修改类别、确认状态、删除误检或填写备注。
6. 点击“导出 Excel”生成台账。

## 注意

- 第一版不做登录、不做长期历史库，结果保存在 `backend\runtime\jobs` 临时任务目录。
- 如果模型权重不存在，后端会在第一次调用模型时提示错误。
- 如果没有 CUDA，后端会自动使用 CPU，但推理速度会慢。
