# ĐỒ ÁN NHẬN DIỆN BIỂN SỐ XE

## 1. Giới thiệu

Chương trình nhận diện biển số xe từ ảnh hoặc video.

Luồng xử lý:

```text
Ảnh/frame → YOLO phát hiện → Cắt biển số → OCR → Chuẩn hóa → Hiển thị
```

YOLOv8n được fine-tune trên dataset biển số xe. EasyOCR đọc ký tự trong vùng
biển số mà YOLO đã tìm được.

## 2. Cấu trúc project

```text
NhanDienBienSo_AI/
├── main.py                   # File chạy chương trình
├── plate_recognition/
│   ├── detector.py           # Phát hiện biển số bằng YOLO
│   ├── ocr.py                # Tiền xử lý ảnh và đọc ký tự
│   ├── pipeline.py           # Xử lý ảnh và video
│   └── types.py              # Dữ liệu dùng chung
├── models/
│   ├── best.pt               # Model đang được sử dụng
│   ├── model_metadata.json   # Thông tin model và confidence
│   └── yolov8n.pt            # Trọng số YOLOv8n ban đầu
├── datasets/license_plate/
│   ├── images/               # Ảnh của dataset
│   ├── labels/               # Nhãn YOLO
│   ├── splits/               # Danh sách train, val và test
│   └── data.yaml             # Cấu hình dataset
├── BienSoXe/                 # Ảnh và video demo
├── runs/
│   ├── training/             # Kết quả train
│   └── evaluation/           # Kết quả đánh giá
├── scripts/                  # Chuẩn bị dữ liệu và đánh giá model
├── tests/                    # Kiểm tra các chức năng
├── .venv-train/              # Môi trường Python
└── requirements.txt          # Danh sách thư viện
```

Khi demo chỉ cần chạy `main.py`. Thư mục `plate_recognition` tách từng bước
xử lý để code dễ đọc. `datasets`, `scripts` và `runs` lưu quá trình tạo và
đánh giá model.

## 3. Cài đặt môi trường

Tạo môi trường Python 3.11 bằng PowerShell:

```powershell
py -3.11 -m venv .venv-train
.\.venv-train\Scripts\python.exe -m pip install --upgrade pip
.\.venv-train\Scripts\python.exe -m pip install -r requirements.txt
```

Môi trường của model hiện tại gồm Python 3.11.7, Ultralytics 8.4.120,
EasyOCR 1.7.2, PyTorch 2.13.0 CPU và OpenCV 5.0.0.

## 4. Chạy chương trình

### Chạy trực tiếp trong VS Code

1. Mở thư mục `NhanDienBienSo_AI`.
2. Mở `main.py` và bấm **Run Python File**.
3. Nhập vào terminal:
   - `xe_01.jpg` để nhận diện ảnh trong `BienSoXe`;
   - `video_1.mp4` để nhận diện video.

Có thể nhập tên file trong `BienSoXe` hoặc dán một đường dẫn đầy đủ. Sau khi
đóng một kết quả, chương trình quay lại phần nhập để có thể thử file khác mà
không phải chạy lại `main.py`.

Kết quả được in trên terminal và hiện trong cửa sổ OpenCV. Dấu **X** hoặc
phím `Esc` đóng cửa sổ hiện tại rồi quay lại phần nhập.

### Chạy bằng lệnh

```powershell
.\.venv-train\Scripts\python.exe main.py ".\BienSoXe\xe_01.jpg"
.\.venv-train\Scripts\python.exe main.py ".\BienSoXe\video_1.mp4"
```

Thay đổi ngưỡng phát hiện:

```powershell
.\.venv-train\Scripts\python.exe main.py ".\BienSoXe\xe_01.jpg" --confidence 0.85
```

Chương trình chỉ hiển thị kết quả trên terminal và cửa sổ nhận diện, không
tạo thêm ảnh, video, crop hoặc JSON sau mỗi lần chạy.

## 5. Kiểm tra biển số trùng lặp

Các biển số đã nhận diện được ghi nhớ trong RAM khi `main.py` đang mở. Lần
đầu gặp một biển số, chương trình báo `Lượt quét mới` và hiển thị `MOI`. Nếu
chuỗi đó xuất hiện lại, chương trình báo `Đã quét trước đó` và hiển thị
`DA QUET`.

Trong video, cùng một biển số xuất hiện liên tục qua nhiều frame
chỉ được tính một lần. Chương trình so sánh đúng chuỗi đã chuẩn hóa, không tự
ghép hai chuỗi gần giống nhau vì có thể làm nhầm hai xe khác nhau.

Lịch sử này không được ghi ra file. Khi tắt chương trình, bộ nhớ được xóa;
lần mở `main.py` tiếp theo sẽ bắt đầu một phiên quét mới.

## 6. Dataset

Dataset lấy từ bộ
[VN License Plate trên Kaggle](https://www.kaggle.com/datasets/bomaich/vnlicenseplate/data)
và đã có nhãn YOLO. Các bước xử lý trong project:

1. Kiểm tra cặp ảnh và nhãn tương ứng.
2. Kiểm tra class, số trường và giới hạn tọa độ.
3. Chia lại train, validation và test theo nhóm nguồn clip.
4. Khai báo dataset trong `data.yaml`.
5. Fine-tune YOLOv8n và đánh giá model.

Dataset có một lớp `license_plate`, mã lớp là `0`. Mỗi dòng nhãn YOLO có dạng
`class_id x_center y_center width height`.

Dữ liệu có 498 ảnh. Tập được chia lại như sau:

| Tập | Ảnh | Bounding box |
|---|---:|---:|
| Train | 350 | 598 |
| Validation | 73 | 104 |
| Test | 75 | 78 |

Chia theo nguồn clip giúp các frame gần giống nhau không đồng thời nằm trong
train và test, nhờ đó hạn chế data leakage.

Kiểm tra hoặc tạo lại file split:

```powershell
.\.venv-train\Scripts\python.exe .\scripts\prepare_splits.py
.\.venv-train\Scripts\python.exe .\scripts\prepare_splits.py --write
```

## 7. Train model

`models/yolov8n.pt` là trọng số pretrained dùng làm điểm bắt đầu. Sau khi fine-tune,
checkpoint tốt nhất nằm tại `runs/training/plate_final_v1/weights/best.pt`.
`models/best.pt` là bản model mà `main.py` sử dụng.

Lệnh train model final:

```powershell
.\.venv-train\Scripts\yolo.exe detect train `
  model=".\models\yolov8n.pt" `
  data=".\datasets\license_plate\data.yaml" `
  epochs=30 imgsz=640 batch=4 `
  device=cpu workers=0 patience=10 cache=False `
  project=".\runs\training" name="plate_final_v1" exist_ok=False
```

Model được đặt tối đa 30 epoch. Quá trình dừng ở epoch 27 do early stopping
với `patience=10`; checkpoint tốt nhất nằm ở epoch 17. Thời gian train bằng
CPU khoảng 64 phút.

Kết quả validation của checkpoint tốt nhất:

- Precision: 0.999;
- Recall: 0.981;
- mAP50: 0.995;
- mAP50-95: 0.929.

Confidence mặc định là `0.85` để giảm box nhầm ở logo hoặc tem xe. Hai bản
model có cùng SHA-256 bắt đầu bằng `9d06fcddfde16995f...`, xác nhận
`models/best.pt` đúng là checkpoint của run final.

`runs/training/plate_final_v1` còn có cấu hình train, kết quả từng epoch,
biểu đồ loss, ảnh validation, confusion matrix, `best.pt` và `last.pt`.
Nếu train lại, cần đổi tên run để không ghi đè kết quả cũ.

## 8. Kiểm thử và demo

```powershell
.\.venv-train\Scripts\python.exe -m pip check
.\.venv-train\Scripts\python.exe -B -m unittest discover -s tests -v
.\.venv-train\Scripts\python.exe main.py --help
```

Nên dùng `BienSoXe\xe_01.jpg` để demo ảnh ngoài dataset. Có thể chạy thêm
`datasets\license_plate\images\test\clip4_new_10.jpg` để minh họa tập test.

YOLO đánh giá khả năng tìm vị trí biển số. Kết quả OCR còn phụ thuộc vào độ
nét, góc chụp, kích thước và ánh sáng.

## 9. Git và GitHub

Git lưu lịch sử thay đổi của source code. Các trọng số phụ `*.pt`, ảnh và nhãn
dataset, ảnh/video demo trong `BienSoXe` và toàn bộ `runs` được loại khỏi Git do
dung lượng lớn. Riêng `models/best.pt` được theo dõi có chủ đích để sau khi clone
repository và cài thư viện, chương trình có thể chạy nhận diện ngay. GitHub vẫn
lưu source code, README, cấu hình và split dataset, script, test cùng model này.

`models/yolov8n.pt` chỉ cần khi train lại và không được đưa lên repository.
Dataset đầy đủ có thể tải lại từ Kaggle; các kết quả train cần được lưu riêng
khi bàn giao nếu nhóm muốn giữ lại.

Repository của project:
[trinhnhatha2006-lab/NhanDienBienSo_AI](https://github.com/trinhnhatha2006-lab/NhanDienBienSo_AI).
Repository đang để private. Khi làm nhóm, mỗi thành viên có thể tạo branch,
commit phần mình làm rồi mở pull request để cả nhóm xem lại trước khi gộp code.

## 10. Nội dung cần hiểu

- YOLO tìm vị trí biển số; EasyOCR đọc ký tự trong vùng đã cắt.
- Train dùng để học, validation để chọn model, test để đánh giá cuối cùng.
- `best.pt` có kết quả validation tốt nhất; `last.pt` là epoch cuối.
- Precision cao nghĩa là ít phát hiện nhầm; recall cao nghĩa là ít bỏ sót.
- mAP đánh giá tổng quát chất lượng phát hiện.
- Early stopping dừng train khi model không cải thiện thêm.

## 11. Nguồn tham khảo

- Dataset: [Kaggle VN License Plate](https://www.kaggle.com/datasets/bomaich/vnlicenseplate/data)
- YOLOv8: [Ultralytics](https://docs.ultralytics.com/)
- OCR: [EasyOCR](https://github.com/JaidedAI/EasyOCR)
