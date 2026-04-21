# Cá nhân suy ngẫm (Individual Reflection)

**Họ và tên:** Nguyễn Thị Thanh Huyền
**Vai trò:** Judge & Evaluation Specialist (Member 2)

## 1. Engineering Contribution (Đóng góp kỹ thuật - 15đ)
Trong lab này, tôi chịu trách nhiệm chính trong việc thiết kế và lập trình các module đánh giá (Evaluation) phức tạp:
- **Multi-Judge Consensus Engine:** Tôi đã tự tay lập trình module `LLMJudge` trong `engine/llm_judge.py`. Thay vì dùng một model duy nhất, tôi thiết kế hệ thống gọi song song (async) hai model đầu ngành là `GPT-4o` (OpenAI) và `Claude-3.5-Sonnet` (Anthropic) thông qua API để chấm điểm câu trả lời của RAG.
- **Async Pipeline & Automations:** Để xử lý toàn bộ 50 test cases một cách chớp nhoáng, tôi đã sử dụng thư viện `aiohttp` kết hợp `asyncio.gather()` giúp các request API được thực thi đồng thời. Ngoài ra, tôi thiết lập logic tự động ra quyết định Release Gate dựa trên việc so sánh Delta giữa bản V1 và V2.

## 2. Technical Depth (Chiều sâu kỹ thuật - 15đ)
Trong quá trình phát triển hệ thống, tôi đã áp dụng các khái niệm lý thuyết vào thực tiễn:
- **Cohen's Kappa & Agreement Rate:** Tôi áp dụng hệ số Cohen's Kappa để đo lường "độ đồng thuận thực sự" (loại trừ yếu tố đồng thuận do ngẫu nhiên) giữa hai giám khảo. Nếu Kappa < 0.5 hoặc độ lệch điểm > 1.0, một model thứ 3 (Groq Llama-3) sẽ được gọi tự động để làm trọng tài phân xử (Tie-breaker).
- **Position Bias (Thiên vị vị trí):** Tôi đã phát triển hàm `check_position_bias` để tráo đổi thứ tự input đánh giá của các model nhằm kiểm tra và triệt tiêu xu hướng LLM thiên vị cho thông tin xuất hiện trước.
- **MRR (Mean Reciprocal Rank):** Tôi hiểu sâu sắc rằng MRR bổ sung cho Hit Rate. Hit Rate chỉ cho biết chunk liên quan có được kéo về không, còn MRR đánh giá khả năng đưa chunk đó lên Top đầu. Retrieval rank thấp thì Generation dễ bị trôi ngữ cảnh (Answer Quality kém).
- **Trade-off giữa Chi phí và Chất lượng:** Việc sử dụng Multi-Judge giúp chất lượng đánh giá công tâm hơn nhưng đẩy Cost lên gấp đôi. Logic của tôi tối ưu điều này bằng cách: Nếu 2 judge chính đã đồng thuận cao, ta chốt điểm luôn; chỉ gọi thêm trọng tài khi xảy ra mâu thuẫn để tối ưu chi phí.

## 3. Problem Solving (Giải quyết vấn đề - 10đ)
- **Vấn đề 1 (Pipeline crash do Rate Limit/Network):** Khi gửi đồng thời hàng chục request API cho 50 cases, các hệ thống bên ngoài rất hay bị lỗi hoặc giới hạn rate limit.
  - **Cách giải quyết:** Tôi bọc toàn bộ các API call trong cấu trúc `try-except`, nếu có exception hoặc status code lỗi, hệ thống sẽ tự động gọi hàm `_fallback_score()` (mặc định cho điểm 3.0 và ghi log "Fallback due to error") để đảm bảo luồng benchmark không bao giờ bị sập giữa chừng.
- **Vấn đề 2 (Data Contract phá vỡ hệ thống):** Các model thỉnh thoảng không tuân thủ nghiêm ngặt định dạng JSON mà bọc trong markdown (ví dụ ` ```json {..} ``` `), gây lỗi hàm `json.loads()`.
  - **Cách giải quyết:** Tôi đã tích hợp regex `re.search(r'\{.*\}', content, re.DOTALL)` và các logic `.strip()` để bóc tách và làm sạch raw text trước khi parse, đảm bảo độ ổn định tuyệt đối (100% parse thành công).
