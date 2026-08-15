# Nhận diện biển số xe

Phiên bản này được viết lại theo kiến trúc pipeline: phát hiện biển số bằng YOLO, OCR bằng EasyOCR, rồi lưu ảnh/video đã gắn kết quả. Chương trình nhận ảnh và video, cắt từng biển số phát hiện được, và xuất kết quả vào thư mục `runs/`.

## Cài đặt

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Đặt model YOLO đã huấn luyện tại `models/best.pt`, hoặc truyền đường dẫn model bằng `--model`. Có thể dùng trực tiếp file model cũ của bạn, ví dụ `D:\UTH\ARTIFICIAL INTELLIGENCE\NhanDienBienSoAI\models\best.pt`.

## Chạy

Nhận diện từ ảnh:

```powershell
python main.py ".\BienSoXe\xe_01.jpg" --model "D:\UTH\ARTIFICIAL INTELLIGENCE\NhanDienBienSoAI\models\best.pt"
```

Nhận diện từ video:

```powershell
python main.py ".\BienSoXe\video_xe.mp4" --model "models\best.pt" --preview
```

Nhấn `q` để dừng sớm khi đang bật `--preview`.

## Kết quả

Mỗi lần chạy tạo một thư mục trong `runs/` gồm:

- `annotated_image.jpg` hoặc `annotated_video.mp4`: ảnh/video có bounding box và chuỗi OCR.
- `plates/`: ảnh crop của các biển số phát hiện được.
- `recognitions.json`: danh sách khung, độ tin cậy và chuỗi ký tự OCR.

Tùy chọn hữu ích: `--confidence 0.6` đổi ngưỡng YOLO; `--no-save-plates` không lưu ảnh crop; `--no-ocr` chỉ phát hiện biển số.
