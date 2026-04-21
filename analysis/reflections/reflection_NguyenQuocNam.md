# Cá nhân suy ngẫm (Individual Reflection)

**Họ và tên:** Nguyễn Quốc Nam - ** MSV : 2A202600201 **
**Vai trò:** Data & Retrieval Specialist (Member 1)

## 1. Engineering Contribution (Đóng góp kỹ thuật - 15đ)
Trong lab này, tôi đảm nhận vai trò Data & Retrieval, tập trung vào việc xây dựng hệ thống đánh giá Retrieval và tạo dữ liệu kiểm thử:

- **Challenging Questions Generator:** Tôi đã phát triển module `ChallengingQuestionsGenerator` trong `phase1/challenging_questions.py` để tạo 50+ câu hỏi thách thức phục vụ kiểm tra RAG. Các câu hỏi được thiết kế theo 5 loại: semantic inference, negative constraint, temporal reasoning, comparative analysis, và context-dependent.
- **Golden Dataset Generation:** Tôi cải tiến `data/synthetic_gen.py` để tạo bộ 50 câu hỏi khó với 33 câu hỏi adversarial (đánh lừa hệ thống), mỗi câu có metadata về difficulty, type, và explanation.
- **Chunk Verification System:** Tôi xây dựng `ChunkVerifier` với các hàm tính Hit Rate, MRR, và Precision@k để đánh giá khả năng retrieval của hệ thống RAG.

## 2. Technical Depth (Chiều sâu kỹ thuật - 15đ)
Trong quá trình phát triển, tôi đã áp dụng các khái niệm lý thuyết vào thực tiễn:

- **Hit Rate & MRR:** Tôi hiểu sâu rằng Hit Rate cho biết chunk liên quan có được retrieval hay không, còn MRR (Mean Reciprocal Rank) đánh giá vị trí của chunk đầu tiên được tìm thấy. MRR = 1/position nên rank càng cao, MRR càng thấp.
- **Retrieval Quality vs Answer Quality:** Tôi nhận thức rằng Retrieval là nền tảng - nếu retrieval kém (sai chunk), generation sẽ bị hallucinate. Chunk verification giúp xác định chính xác lỗi ở đâu: ingestion, chunking, hay retrieval.
- **Negative Constraints:** Tôi thiết kế câu hỏi "NOT suitable" yêu cầu hệ thống phải filter thông tin, đây là edge case quan trọng để test semantic understanding vs keyword matching.
- **Temporal Reasoning:** Các câu hỏi về "pre-2020" vs "post-2020" yêu cầu hệ thống hiểu chronology và đánh giá sự tiến hóa của phương pháp.

## 3. Problem Solving (Giải quyết vấn đề - 10đ)
- **Vấn đề 1 (Test failures - LLMJudge not defined):** Các test trong phase1 file gọi LLMJudge nhưng thiếu import.
  - **Cách giải quyết:** Tôi thêm `from engine.llm_judge import LLMJudge` vào test imports và fix các assertion không phù hợp với implementation thực tế.
- **Vấn đề 2 (JSON format issues):** Hàm save_questions tạo ra JSON thay vì JSONL format.
  - **Cách giải quyết:** Tôi cập nhật test để handle cả hai format JSON và JSONL.
- **Vấn đề 3 (Async handling in tests):** Một số test cần asyncio nhưng thiếu import.
  - **Cách giải quyết:** Thêm `import asyncio` vào test files.

## 4. Testing Results
- **Phase 1 tests:** 8/8 challenging questions generator tests passed
- **Metric tests:** 8/8 ChunkVerifier metrics tests passed  
- **Overall:** Most tests passing, some edge cases being refined

## 5. Learning Outcomes
Qua lab này, tôi hiểu sâu hơn về:
- Sự quan trọng của retrieval trong hệ thống RAG - đây là "nền móng" quyết định chất lượng answer
- Cách thiết kế adversarial questions để phát hiện điểm yếu của hệ thống
- Trade-off giữa different retrieval strategies (BM25 vs semantic search)
- Ý nghĩa của "garbage in, garbage out" - nếu retrieval sai chunk, answer sẽ hallucinate

---

*Ngày cập nhật: 2026-04-21*