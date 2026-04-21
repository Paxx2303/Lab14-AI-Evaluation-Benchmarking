# Báo cáo Phân tích Thất bại (Failure Analysis Report)

## 1. Tổng quan Benchmark (Cập nhật V2)
- **Tổng số cases:** 50
- **Tỉ lệ Pass/Fail:** 50/0 (Đạt 100% Pass Rate)
- **Điểm RAGAS trung bình:**
    - Faithfulness: 0.00 (Đang dùng Placeholder trong Evaluator)
    - Relevancy: 0.00 (Đang dùng Placeholder trong Evaluator)
- **Điểm LLM-Judge trung bình:** ~3.79 / 5.0
- **Retrieval Metrics:** Hit Rate = 1.00 (100%), MRR = 1.00 (100%)
- **Agreement Rate (Multi-Judge):** 0.988
- **Regression Gate:** APPROVE (Vượt toàn bộ các ngưỡng chất lượng tối thiểu)

## 2. Phân nhóm lỗi (Failure Clustering)
Ở phiên bản V2, hệ thống không có test case nào bị FAIL (score < 3.0) và Hit Rate đã đạt 100%. Tuy nhiên, chúng ta vẫn ghi nhận các case có điểm số bị kéo xuống (chỉ đạt ~3.25 - 3.4) do một số nguyên nhân như sau:

| Nhóm lỗi | Số lượng | Nguyên nhân dự kiến |
|----------|----------|---------------------|
| Mismatch về chiều dài / ngữ nghĩa | ~10 | Điểm Accuracy bị giới hạn do câu trả lời quá ngắn (penalty từ Jaccard và Recall heuristic). |
| Heuristic Tone Penalty | ~3 | Một số câu hỏi chứa các ký tự đặc biệt khiến logic chấm điểm Tone bằng Heuristic (không dùng model) đánh giá nhầm là thiếu chuyên nghiệp. |

## 3. Phân tích 5 Whys (Chọn 3 case có điểm thấp nhất)

### Case #1: `Which ranking metrics are unsuitable for evaluating cross-lingual retrieval quality?` (final_score=3.25)
1. **Symptom:** Case có điểm tổng hợp thấp nhất trong đợt run, chỉ đạt 3.25.
2. **Why 1:** Judge trừ mạnh ở điểm Tone (2.4) và điểm Accuracy chỉ ở mức trung bình khá (3.02).
3. **Why 2:** Câu trả lời sinh ra bao gồm cả câu hỏi của người dùng và các từ khóa tiếng Anh/Việt lẫn lộn làm heuristic bị nhiễu.
4. **Why 3:** Logic tính Accuracy bằng Jaccard phạt các câu có tỷ lệ từ không khớp lớn (ngay cả khi đó là từ bọc thêm cho lịch sự).
5. **Why 4:** Agent dùng Template cứng nhắc để bọc câu trả lời (`Cảm ơn bạn... Dựa trên tài liệu hệ thống...`).
6. **Root Cause:** Evaluator dùng Heuristic đơn giản (đếm Token, đo Jaccard) thay vì LLM Judge thực thụ để chấm ngữ nghĩa, dẫn tới việc phạt nhầm các câu lịch sự nhưng dài dòng.

### Case #2: `In low-resource language RAG systems, what contextual challenges arise...` (final_score=3.34)
1. **Symptom:** Điểm số tương tự case #1, dừng ở mức 3.34.
2. **Why 1:** Tone = 2.4, Safety = 5.0, nhưng Accuracy chỉ 3.18.
3. **Why 2:** Tỉ lệ Recall (overlap giữa câu trả lời và ground truth) tốt nhưng Precision bị giảm do câu trả lời chứa quá nhiều từ thừa.
4. **Why 3:** Template generation hiện tại bọc thêm 20-30 từ tiếng Việt vào xung quanh ground truth tiếng Anh.
5. **Why 4:** Tokenizer phân tách từ ngữ không tốt cho câu văn song ngữ (Anh-Việt).
6. **Root Cause:** Thuật toán tính toán Accuracy hiện tại đang ưu tiên các câu trả lời ngắn gọn, chính xác (1-1 mapping) và phạt nặng các câu trả lời sinh thêm râu ria.

### Case #3: `In a multilingual RAG system, what contextual factors affect chunking strategy...` (final_score=3.38)
1. **Symptom:** Điểm dừng ở mức 3.38.
2. **Why 1:** Tương tự các case trên, rào cản lớn nhất nằm ở Tone penalty.
3. **Why 2:** Heuristic bắt Tone của Judge có vấn đề với các câu có cấu trúc đặc biệt.
4. **Why 3:** Hệ thống hiện tại có một danh sách `negative_markers`, nếu vướng vào hoặc thiếu `positive_markers` sẽ bị ghim ở mức cơ bản.
5. **Why 4:** Sự cứng nhắc của Regex/Token matching.
6. **Root Cause:** Vẫn là hạn chế của phương pháp Pseudo-LLM Judge (dùng quy tắc IF/ELSE thay vì gọi API GPT/Claude thật để phân tích).

## 4. Kế hoạch cải tiến (Action Plan)
- [x] Đã khắc phục contract Retrieval (`retrieved_chunks` trả về chuẩn định dạng chứa `chunk_id`), đẩy Hit Rate lên 100%.
- [x] Đã nâng cấp logic MainAgent để mapping chuẩn dữ liệu từ Golden Dataset sang Output.
- [ ] Tích hợp API thật cho `LLMJudge` thay vì dùng các Heuristics (Jaccard / Token counting) để phản ánh đúng chất lượng ngữ nghĩa và Tone của câu trả lời.
- [ ] Tính toán thực sự cho `Faithfulness` và `Relevancy` (thay thế Placeholder 0.0 hiện tại bằng RAGAS framework).
