🎬 MovieLens-GraphNeuralNetwork

Graph Neural Network project on MovieLens 1M dataset.

این پروژه با هدف ساخت، آموزش و ارزیابی یک مدل **Graph Neural Network (GNN)** روی دیتاست **MovieLens 1M** طراحی شده است.

---

📁 ساختار پوشه‌ها و فایل‌ها

🔹 .vscode/
تنظیمات محیط توسعه VS Code برای فرمت خودکار، linting و انتخاب مفسر پایتون.

🔹 data/

- `raw/`: داده‌های خام و اصلی MovieLens.
- `processed/`: داده‌هایی که پیش‌پردازش شده‌اند و آماده‌ی ساخت گراف هستند.

🔹 notebooks/
فایل‌های Jupyter برای تحلیل اولیه داده‌ها و تست مدل.

🔹 src/
کدهای اصلی پایتون:

- `data_loader.py`: بارگذاری و آماده‌سازی داده‌ها.
- `build_graph.py`: ساخت گراف (Node, Edge, Feature).
- `model.py`: تعریف مدل GNN (مثلاً GCN یا GAT).
- `train.py`: حلقه‌ی آموزش، ثبت لاگ و ذخیره مدل.
- `evaluate.py`: محاسبه‌ی دقت، F1، ROC و سایر معیارها.
- `utils.py`: توابع کمکی مثل ذخیره نمودارها، مدیریت مسیرها و ...

🔹 outputs/

- `models/`: مدل‌های ذخیره‌شده پس از آموزش.
- `plots/`: نمودارهای loss/accuracy و سایر خروجی‌های گرافیکی.
- `logs/`: لاگ‌های آموزش برای TensorBoard.
- `metrics/`: نتایج عددی مدل (مثل دقت یا Confusion Matrix).

🔹 configs/

- `config.yaml`: پیکربندی قابل تنظیم شامل پارامترهای مدل، مسیر داده و تنظیمات آموزش.

🔹 tests/
تست‌های واحد برای اطمینان از درستی توابع اصلی.

🔹 requirements.txt
پکیج‌های مورد نیاز برای اجرای پروژه.

🔹 run.py
نقطه ورود برای آموزش و ارزیابی مدل.

🔹 .gitignore
نادیده گرفتن فایل‌های خروجی، محیط مجازی و فایل‌های موقتی.

---

🚀 شروع سریع

```bash
 نصب وابستگی‌ها
pip install -r requirements.txt

 اجرای پروژه
python run.py
```
