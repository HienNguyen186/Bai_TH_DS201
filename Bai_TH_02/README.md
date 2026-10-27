# BaiTH02 — VinaFood21

## Notebook chính

`bai_th02.ipynb` chứa cả bốn câu trong `assignments.ipynb`. Phần đầu dùng chung cho import, kiểm tra dataset, chia train/validation, transform và các hàm train/test. Sau đó, **mỗi mô hình có các ô riêng: cấu hình → Train → Test → Kết quả**. Bạn có thể chạy một mô hình mà không phải chạy ba mô hình khác.

| Câu | Mô hình | Phần code riêng |
|---|---|---|
| 1 | LeNet-5 | Hai convolution, pooling, fully connected |
| 2 | GoogLeNet | Chín Inception blocks tự xây dựng |
| 3* | ResNet-18 | Residual blocks theo bốn stage `[2,2,2,2]` |
| 4* | ResNet50 Hugging Face | Fine-tune `microsoft/resnet-50` pretrained |

Dấu `*` theo đề là bài mở rộng. `DS201_Lab2_Completed_PyTorch.ipynb` là notebook khác, hoàn thành các ô `TODO` của `DS201_Lab2_Student_PyTorch.ipynb`. Các file `Bai1_LeNet.ipynb` đến `Bai4_ResNet50_HF.ipynb` là bản chạy riêng trước khi gộp; notebook cần dùng hiện nay là `bai_th02.ipynb`.

## Dataset chung

Dataset ở `BaiTH02/VinaFood21` có **10.044 ảnh train**, **6.682 ảnh test** và **21 lớp**. Với seed 42, notebook tách khoảng 10% ảnh *trong từng lớp* từ train: **9.038 train / 1.006 validation / 6.682 test**. Cả bốn mô hình dùng cùng chỉ số chia tập. Test không dùng để chọn epoch; epoch tốt nhất được chọn bằng validation macro-F1.

Câu 1 đọc ảnh gốc với JPEG draft. Câu 2–4 dùng cache ảnh cạnh dài tối đa 320 pixel tại `VinaFood21_cache320/` để tăng tốc đọc ảnh. Cache giữ nguyên nhãn và split, không sửa ảnh gốc. Nếu chuyển máy, đặt dataset gốc vào `VinaFood21/`; ô dataset chung sẽ tạo lại cache khi cần.

## Chạy riêng từng mô hình đủ 20 epoch

1. Mở `bai_th02.ipynb` trong Jupyter/VS Code và chạy các ô chung từ đầu đến hết **“2. Hàm train và test dùng chung”**.
2. Đến phần mô hình cần chạy, trong ô **Cấu hình riêng** giữ `EPOCHS_<TÊN> = 20`, đổi cả `TRAIN_<TÊN>` và `TEST_<TÊN>` thành `True`. Ví dụ Bài 1:

   ```python
   EPOCHS_LENET = 20
   TRAIN_LENET = True
   TEST_LENET = True
   ```

3. Chạy các ô của mô hình đó theo thứ tự **mô hình → cấu hình → Train → Test → Kết quả**. Ô Train in tiến độ từng epoch. Ô Test đánh giá mô hình tại epoch có validation macro-F1 tốt nhất. Ô Kết quả hiển thị accuracy, macro-precision, macro-recall, macro-F1 và báo cáo theo 21 lớp.
4. Muốn chạy bài khác, đến phần tương ứng và bật hai cờ riêng của bài đó. Không cần chạy lại các ô dataset chung nếu kernel chưa khởi động lại.

Tên cờ của các bài còn lại là `TRAIN_GOOGLENET`/`TEST_GOOGLENET`, `TRAIN_RESNET18`/`TEST_RESNET18`, `TRAIN_RESNET50_HF`/`TEST_RESNET50_HF`. Mặc định các cờ là `False` để tránh vô tình train hàng giờ khi bấm Run All.

Notebook **không ghi file kết quả mới, không xuất confusion matrix**. Mô hình tốt nhất được giữ trong RAM của kernel. Nếu khởi động lại kernel, bạn cần train lại trước khi test. Hãy lưu notebook sau khi chạy để giữ output hiển thị.

## Kết quả tham khảo của lần chạy 3 epoch trước

Các số sau đã được hiển thị sẵn ở ô Kết quả của từng mô hình khi chưa train lại. Chúng **không phải** kết quả của lần chạy 20 epoch mới.

- **LeNet-5:** accuracy 0,2360; macro-precision 0,2068; macro-recall 0,2108; macro-F1 0,1750.
- **GoogLeNet:** accuracy 0,1202; macro-precision 0,0454; macro-recall 0,0996; macro-F1 0,0519.
- **ResNet-18:** accuracy 0,3002; macro-precision 0,3293; macro-recall 0,2879; macro-F1 0,2535.
- **ResNet50 pretrained:** accuracy 0,6937; macro-precision 0,7021; macro-recall 0,7021; macro-F1 0,6853.

GoogLeNet tự xây dựng chưa có hai auxiliary classifiers của kiến trúc gốc. Cả ba mô hình học từ đầu mới chạy 3 epoch nên kết quả tham khảo còn thấp. Trong lần chạy trước trên NVIDIA T1200 4 GB, FP16 tạo loss `NaN` ở GoogLeNet và ResNet-18; notebook dùng FP32 và cắt gradient cho câu 2–4, còn LeNet dùng mixed precision.

## Ý nghĩa code và kiến thức cần nắm

| Khối | Ý nghĩa | Kiến thức nên học |
|---|---|---|
| `ImageFolder`, `DataLoader`, chia validation | Đọc ảnh theo thư mục lớp, tạo batch, giữ test riêng | Nhãn số, shuffle, seed, tránh rò rỉ dữ liệu |
| `make_loaders` | Dùng cùng chỉ số chia tập nhưng resize theo kích thước của từng mô hình | Transform, augmentation, normalize, tensor `[N,C,H,W]` |
| `LeNet5` | Trích đặc trưng bằng conv/pooling rồi phân loại bằng Linear | Kernel, stride, ReLU, flatten |
| `Inception`, `GoogLeNet` | Nhiều nhánh song song, nối theo chiều channel | Padding, conv 1×1, `torch.cat(dim=1)`, global average pooling |
| `BasicBlock`, `ResNet18` | Cộng nhánh chính với shortcut | Residual connection, BatchNorm, downsampling |
| `HFResNet50` | Tải pretrained, thay classifier 21 lớp, fine-tune toàn mạng | Transfer learning, learning rate thấp |
| `run_epoch`, `fit_in_memory` | Tính loss, backpropagation, Adam; giữ trọng số tốt nhất trong RAM | `model.train()`/`eval()`, gradient, validation, overfitting |
| `test_in_memory`, `show_metrics` | Đánh giá test và in số liệu theo lớp | Accuracy, precision, recall, F1, TP/FP/FN |

Đề ghi “4 độ đo” nhưng chỉ nêu precision, recall và F1; code báo cáo thêm **accuracy**. Precision, recall, F1 được tính theo **macro average** để mỗi lớp đóng góp như nhau.

## Nộp bài

Theo `assignments.ipynb`, commit/push code lên GitHub rồi tạo `<MSSV>.txt` chứa link repo để nộp. `.gitignore` loại dataset, cache ảnh và checkpoint cũ khỏi GitHub.
