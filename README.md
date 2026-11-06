# Student Guide: Nền tảng LLM API (Step-by-Step Walkthrough)

## Bài giải đã hoàn thành

Bài giải nằm ở [`solution-code/solution.py`](solution-code/solution.py), là đường dẫn bộ chấm ưu tiên. [`starter-code/template.py`](starter-code/template.py) cũng đã được điền đầy đủ. Cả hai giữ nguyên chữ ký hàm của đề bài.

- Task 1–3: gọi OpenAI, Gemini và Anthropic, trả về văn bản, thời gian gọi API và số token.
- Task 4: so sánh GPT-4o, GPT-4o Mini và Gemini Flash, tính chi phí theo `PRICING_1M_TOKENS` của đề bài.
- Task 5: chatbot Gemini streaming, lưu 3 lượt hội thoại hoàn chỉnh gần nhất; `quit`, `exit`, EOF hoặc Ctrl+C ở ô nhập để thoát.
- Bonus A–C: retry với exponential backoff, so sánh nhiều prompt và xuất bảng Markdown.

Chạy từ thư mục gốc repo, sau khi tạo và kích hoạt `.venv` theo hướng dẫn bên dưới:

```bash
python -m pip install -r requirements.txt
python -m pytest tests/ -v
python solution-code/solution.py
```

Các bài kiểm tra giả lập toàn bộ API, không cần key và không gọi dịch vụ thật. Chạy script trực tiếp sẽ gọi API thật: sao chép `.env.example` thành `.env`, điền `OPENAI_API_KEY` và `GEMINI_API_KEY`. Chỉ cần thêm `ANTHROPIC_API_KEY` khi gọi `call_anthropic`. Script tự đọc `.env` ở thư mục gốc; khi import các hàm vào ứng dụng khác, hãy thiết lập biến môi trường hoặc gọi `load_dotenv()` trước.

Giá và model mặc định được giữ theo đề bài để tương thích bộ chấm; chi phí tính ra là ước lượng của bài lab, không phải báo giá hiện hành hoặc hóa đơn thực tế (ví dụ chưa tính token suy luận của Gemini). Khi chạy thật, truyền `model=` phù hợp với model mà tài khoản của bạn còn truy cập được.

Tài liệu SDK đã đối chiếu: [OpenAI Chat Completions](https://developers.openai.com/api/reference/python/resources/chat/subresources/completions/methods/create), [Google Gen AI Python](https://googleapis.github.io/python-genai/), [Anthropic Messages](https://platform.claude.com/docs/en/api/python/messages/create). Bài giải dùng `google-genai`, không cần SDK `google-generativeai` cũ.

> [!NOTE]
> Tài liệu này được thiết kế để hỗ trợ bạn hoàn thành bài Lab 01 một cách nhanh chóng nhất. Hãy sử dụng nó làm cẩm nang khi triển khai các hàm trong `template.py`.

---

## 🛠️ 1. Khởi tạo Môi trường & Thiết lập API Keys

### 🔹 1.1 Khởi tạo môi trường ảo (venv) & Cài đặt thư viện
Để tránh xung đột thư viện giữa các dự án, bạn nên sử dụng môi trường ảo (`.venv`):

1. **Khởi tạo Virtual Environment (`.venv`):**
   * **Windows/macOS/Linux**:
     ```bash
     python -m venv .venv
     ```

2. **Kích hoạt môi trường ảo (Activate):**
   * **Windows (PowerShell)**:
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
   * **Windows (CMD)**:
     ```cmd
     .venv\Scripts\activate.bat
     ```
   * **macOS/Linux**:
     ```bash
     source .venv/bin/activate
     ```

3. **Cài đặt thư viện từ [requirements.txt](requirements.txt):**
   ```bash
   pip install -r requirements.txt
   ```

---

### 🔹 1.2 Hướng dẫn lấy API Keys

* **Google Gemini API Key (GEMINI_API_KEY):**
  1. Truy cập trang web [Google AI Studio](https://aistudio.google.com/).
  2. Đăng nhập bằng tài khoản Google của bạn.
  3. Bấm vào nút **"Get API key"** (ở menu bên trái hoặc góc trên).
  4. Chọn **"Create API key"** và tạo khóa cho dự án mới hoặc dự án hiện có.
  5. Sao chép API key đã tạo (nó sẽ có dạng bắt đầu bằng `AIzaSy...`).

* **OpenAI API Key (OPENAI_API_KEY):**
  1. Truy cập [OpenAI API Keys Platform](https://platform.openai.com/api-keys).
  2. Đăng nhập/Đăng ký tài khoản OpenAI của bạn.
  3. Bấm vào nút **"Create new secret key"**.
  4. Đặt tên gợi nhớ cho key (ví dụ: `VinUni_Lab1`) và chọn quyền hạn (mặc định là All).
  5. Sao chép API key đã tạo (nó sẽ có dạng bắt đầu bằng `sk-proj-...`).
  > [!WARNING]
  > Key OpenAI chỉ hiển thị duy nhất **một lần** ngay khi tạo. Hãy lưu trữ key an toàn.

---

### 🔹 1.3 Thiết lập Biến môi trường trực tiếp (Environment Variables)

Bạn có thể thiết lập biến môi trường trực tiếp từ terminal (áp dụng cho phiên làm việc hiện tại) bằng các lệnh sau:

* **Windows (PowerShell):**
  ```powershell
  $env:OPENAI_API_KEY="sk-proj-your-openai-key-here"
  $env:GEMINI_API_KEY="AIzaSy-your-gemini-key-here"
  ```

* **Windows (CMD):**
  ```cmd
  set OPENAI_API_KEY=sk-proj-your-openai-key-here
  set GEMINI_API_KEY=AIzaSy-your-gemini-key-here
  ```

* **macOS/Linux (Terminal):**
  ```bash
  export OPENAI_API_KEY="sk-proj-your-openai-key-here"
  export GEMINI_API_KEY="AIzaSy-your-gemini-key-here"
  ```

> [!TIP]
> Để tránh việc phải nhập lại lệnh thiết lập mỗi khi tắt terminal, bạn có thể tạo một file `.env` đặt ở thư mục gốc của project (sử dụng mẫu từ [.env.example](.env.example)). Script bài giải đã dùng `python-dotenv` để load tự động khi chạy trực tiếp.

---

## 🚀 2. Lộ trình triển khai (Recommended Flow)

Để hoàn thành bài lab một cách khoa học, hãy đi theo trình tự sau:

    A[Task 1: call_openai] --> B[Task 2: call_gemini]
    B --> C[Task 3: call_anthropic]
    C --> D[Task 4: compare_models]
    D --> E[Task 5: streaming_chatbot]
    E --> F[Bonus A: retry_with_backoff]
    F --> G[Bonus B/C: batch & table]


---

## 🛠️ 3. Hướng dẫn chi tiết từng nhiệm vụ

### 🔹 Task 1: `call_openai` (OpenAI API)
* **SDK cần dùng:** `from openai import OpenAI`
* **Cách khởi tạo:** `client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))`
* **Hàm gọi:** `client.chat.completions.create(...)`
* **Tham số:**
  * `model`: chuỗi tên model (`gpt-4o` hoặc `gpt-4o-mini`).
  * `messages`: dạng danh sách hội thoại `[{"role": "user", "content": prompt}]`.
  * `temperature`, `top_p`, `max_tokens`.
* **Đo thời gian (Latency):** Dùng thư viện `time`:
  ```python
  start = time.time()
  # call API
  latency = time.time() - start
  ```
* **Lấy Token Usage:**
  ```python
  input_tokens = response.usage.prompt_tokens
  output_tokens = response.usage.completion_tokens
  ```

---

### 🔹 Task 2: `call_gemini` (Google Gemini 2.5) — Trọng tâm chính!
Chúng ta sẽ sử dụng SDK mới nhất của Google (`google-genai`):
* **SDK cần dùng:** `from google import genai` và `from google.genai import types`
* **Cách khởi tạo:** `client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))`
* **Hàm gọi:** `client.models.generate_content(...)`
* **Cấu hình tham số (Configuration):**
  ```python
  config = types.GenerateContentConfig(
      temperature=temperature,
      top_p=top_p,
      max_output_tokens=max_tokens
  )
  response = client.models.generate_content(
      model=model,
      contents=prompt,
      config=config
  )
  ```
* **Lấy Token Usage:**
  ```python
  input_tokens = response.usage_metadata.prompt_token_count
  output_tokens = response.usage_metadata.candidates_token_count
  ```

---

### 🔹 Task 5: `streaming_chatbot` (Gemini Streaming + History)
Streaming giúp tạo trải nghiệm người dùng mượt mà giống như ChatGPT (chữ chạy đến đâu hiển thị đến đó).
* **Streaming logic:**
  ```python
  response_stream = client.models.generate_content_stream(
      model="gemini-2.5-flash",
      contents=formatted_history
  )
  for chunk in response_stream:
      print(chunk.text, end="", flush=True)
  ```
* **Lịch sử hội thoại (History Management):**
  Giới hạn lịch sử ở 3 lượt hội thoại gần nhất (tương đương 6 tin nhắn).
  ```python
  # Cắt lịch sử nếu quá dài
  history = history[-6:]
  ```

---

## 🆘 4. Giải quyết các lỗi thường gặp (Troubleshooting)

### 🔴 Lỗi `ModuleNotFoundError: No module named 'google'`
* **Nguyên nhân:** Chưa cài đặt thư viện Google GenAI SDK.
* **Cách khắc phục:** Chạy `pip install google-genai` trong terminal của bạn.

### 🔴 Lỗi `AuthenticationError` hoặc `API Key Not Found`
* **Nguyên nhân:** API key chưa được thiết lập vào biến môi trường.
* **Cách khắc phục:**
  * **MacOS/Linux:** `export GEMINI_API_KEY="AIzaSy..."`
  * **Windows (PowerShell):** `$env:GEMINI_API_KEY="AIzaSy..."`
  * **Windows (CMD):** `set GEMINI_API_KEY=AIzaSy...`

### 🔴 Lỗi `RateLimitError`
* **Nguyên nhân:** Bạn gửi quá nhiều request trong một khoảng thời gian ngắn trên tài khoản Free.
* **Cách khắc phục:** Triển khai **Task Bonus A (`retry_with_backoff`)** để tự động thử lại sau vài giây!

---
> **💡 Vibe Coding Tip:** Hãy tận dụng Cursor hoặc Copilot bằng cách cung cấp cho nó ngữ cảnh của file `student_guide.md` và `template.py` để viết code nháp nhanh chóng, sau đó kiểm tra và hoàn thiện bằng tay!
