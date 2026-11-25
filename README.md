# Discord Gemini Bot (Python + Online 24/7)

Bot Discord viết bằng **Python (`discord.py`)**, tích hợp **Gemini AI (`google-genai` / `gemini-4-hydrogen`)** và **HTTP health-check server (`aiohttp`)** để chạy ổn định 24/7 trên Render, Railway, Fly.io hoặc Docker.

## Tính năng chính

- **Slash Commands**:
  - `/ping` — Kiểm tra trạng thái online và độ trễ (latency) của bot.
  - `/info` — Xem thông tin bot, thời gian uptime, số server và model AI đang dùng.
  - `/ask <câu hỏi> [image] [private]` — Hỏi Gemini AI trực tiếp, có thể đính kèm ảnh (PNG/JPG/WEBP/GIF) và chế độ chỉ mình bạn thấy.
  - `/reset` — Xóa ngữ cảnh hội thoại AI của bạn trong kênh hiện tại để bắt đầu cuộc trò chuyện mới.
  - `/help` — Hướng dẫn sử dụng các lệnh.
- **Chat tự nhiên & Xem ảnh qua Mention / Reply / DM**:
  - Tag `@TenBot <câu hỏi>` kèm ảnh, **Reply vào một tin nhắn có ảnh** rồi tag `@TenBot`, hoặc nhắn trực tiếp (DM) để bot xem ảnh và trò chuyện liên tục nhiều lượt.
- **Sẵn sàng deploy 24/7**:
  - Tích hợp sẵn HTTP server tại `/` và `/health` trên cổng `$PORT` (mặc định `8080`) để các nền tảng PaaS (Render, Railway, Koyeb...) kiểm tra health-check.

---

## Bước 1: Lấy Discord Bot Token & mời bot vào server

1. Truy cập [Discord Developer Portal](https://discord.com/developers/applications) → **New Application**.
2. Vào mục **Bot**:
   - Nhấn **Reset Token** để lấy `DISCORD_TOKEN` (lưu lại, không chia sẻ công khai).
   - Bật **Message Content Intent** trong phần *Privileged Gateway Intents* (cần thiết để bot đọc được tin nhắn khi được `@mention` hoặc DM).
3. Vào mục **OAuth2 → URL Generator**:
   - Scopes: chọn `bot` và `applications.commands`.
   - Bot Permissions: chọn `Send Messages`, `Read Message History`, `Embed Links`, `Use Slash Commands`.
   - Mở URL được tạo để mời bot vào server Discord của bạn.

## Bước 2: Lấy Gemini API Key (để dùng AI Chat)

1. Truy cập [Google AI Studio](https://aistudio.google.com/apikey).
2. Tạo API key và lưu lại làm `GEMINI_API_KEY`.

---

## Bước 3: Chạy thử trên máy (Local)

```bash
# 1. Tạo file .env từ mẫu và điền DISCORD_TOKEN, GEMINI_API_KEY
cp .env.example .env

# 2. Cài đặt thư viện (khuyên dùng uv)
uv sync

# 3. Chạy bot
uv run python main.py
```

*(Hoặc dùng `pip`: `pip install -r requirements.txt && python main.py`)*

---

## Bước 4: Đưa bot lên mạng chạy Online 24/7

### Cách 1: Deploy lên Render (có sẵn `render.yaml`)
1. Đẩy thư mục dự án lên GitHub.
2. Vào [Render](https://render.com) → **New → Blueprint** (hoặc **Web Service**) và chọn repo của bạn.
3. Render sẽ tự đọc `render.yaml` (`pip install -r requirements.txt` và `python main.py`, health check `/health`).
4. Trong phần **Environment Variables**, thêm:
   - `DISCORD_TOKEN`
   - `GEMINI_API_KEY`
5. *(Mẹo giữ gói Free không bị ngủ)*: Có thể dùng UptimeRobot ping URL `https://<ten-app>.onrender.com/health` mỗi 5 phút.

### Cách 2: Deploy lên Railway / Koyeb / Fly.io
- Dự án có sẵn `Procfile` (`web: python main.py`) và `Dockerfile`.
- Chỉ cần kết nối repo và thêm biến môi trường `DISCORD_TOKEN`, `GEMINI_API_KEY`.

### Cách 3: Chạy bằng Docker trên VPS
```bash
docker build -t discord-gemini-bot .
docker run -d --name discord-bot --restart unless-stopped --env-file .env -p 8080:8080 discord-gemini-bot
```
