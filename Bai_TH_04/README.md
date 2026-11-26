# Bài thực hành 04 — dịch máy Anh–Việt

`assignments.ipynb` triển khai ba mô hình theo `Lab 4.docx`:

1. Encoder–Decoder LSTM không attention.
2. Encoder–Decoder với Bahdanau attention.
3. Encoder–Decoder với Luong global-general attention và input-feeding.

Notebook dùng small-PhoMT từ [thư mục dữ liệu của môn học](https://drive.google.com/drive/folders/186OAOuSEYEDVcry7WP5UBdqECXo26QAb). Dữ liệu được kiểm SHA-256 và giữ trong `data/`; thư mục này cùng `artifacts/` đã được loại khỏi Git bằng `.gitignore`. Notebook có cấu hình Phần C: 3 lớp LSTM, hidden/embedding 256, batch 32, Adam 1e-3, tối đa 10 epoch, ngưỡng 50 token và toàn bộ train/dev/test.

Mở notebook trong thư mục `BaiTH04` rồi chạy từ đầu đến cuối. Có thể chạy bằng `python run_assignment.py` để lưu output sau từng ô và tiếp tục từ checkpoint sau khi gián đoạn. Báo cáo được tạo từ kết quả thật bằng `python make_report.py` sau khi notebook hoàn tất.

Các kết quả nộp gồm `assignments.ipynb` với output và `BaiTH04_BaoCao.pdf`. Checkpoint, lịch sử và dữ liệu nằm cục bộ trong `artifacts/` và `data/`.
