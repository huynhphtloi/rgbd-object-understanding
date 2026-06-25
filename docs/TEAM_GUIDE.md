# TODO
---

## Tú — Train & Push model (chạy trên Google Colab, bật GPU)

- Mở `notebooks/colab_full_pipeline.ipynb` trên Colab → Runtime = GPU (T4), mount Google Drive để giữ weights/outputs
- Chạy lần lượt các cell: cài deps → tải & convert OCID → train YOLO-seg → đánh giá → lưu kết quả demo
- Sau train, weights nằm ở `outputs/models/<run>/weights/best.pt`
- Trong notebook, login Hub rồi push: `!huggingface-cli login` → `!huggingface-cli upload your-username/rgbd-object-understanding-yolo-seg outputs/models/<run>/weights/best.pt best.pt --repo-type model`
- Báo cho Vinh: `repo_id` + tên file `best.pt`

---

## Vinh — Dựng Streamlit app (chụp bằng camera iPhone để test)

- Tạo `src/app/streamlit_app.py` dùng `st.camera_input()` để chụp ảnh từ camera iPhone (mở web trên điện thoại)
- Load model từ Hub: `hf_hub_download(repo_id="your-username/rgbd-object-understanding-yolo-seg", filename="best.pt")` → trỏ vào `config/demo.yaml` mục `weights`
- Ảnh chỉ có RGB (không có depth) → ước lượng depth bằng model mono-depth (vd MiDaS/Depth-Anything) rồi đưa vào `Pipeline.run(...)`
- Hiển thị các tính năng: masks, depth, kích thước (W/H/D cm), quan hệ không gian, JSON
- Chạy: `streamlit run src/app/streamlit_app.py` → mở trên iPhone cùng wifi để test camera
- (Tuỳ chọn) Deploy lên Hugging Face Spaces (SDK = Streamlit) để demo online

---

## Tóm tắt

- **Tú:** train → `best.pt` → `huggingface-cli upload` → báo `repo_id` cho Vinh.
- **Vinh:** Streamlit + camera iPhone → load `best.pt` từ Hub → demo các tính năng.
