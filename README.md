# 🤖 Twitter Bot - Crypto Insider (Enhanced v4)

Bot Twitter tự động đăng bài về các dự án tiền điện tử với phong cách tự nhiên như người thật, sử dụng trí tuệ nhân tạo Gemini/OpenAI để tạo nội dung.

## 🔑 Yêu cầu

### API Keys (bắt buộc)
1. **Twitter API Keys**:
   - `TWITTER_API_KEY`
   - `TWITTER_API_SECRET`
   - `TWITTER_ACCESS_TOKEN`
   - `TWITTER_ACCESS_SECRET`
   - `TWITTER_BEARER_TOKEN`

2. **AI API Key** (1 trong 2):
   - `GEMINI_API_KEY` (Ưu tiên)
   - `OPENAI_API_KEY` (Dự phòng)

### Thư viện Python
```bash
pip install requests schedule python-dotenv
🚀 Tính năng chính
Tạo bài đăng tự nhiên:

12 bài/ngày (8AM - 12PM giờ EU)

Đa dạng định dạng: ngắn, trung bình, dài, thread

9 loại nội dung khác nhau

Quản lý thông minh:

Theo dõi lịch sử bài đăng (tweet_history.json)

Phân tích hiệu suất tweet (tweet_analytics.json)

Tự động trả lời mention

Công nghệ AI:

Ưu tiên Gemini 1.5 Pro (1M token context)

Dự phòng OpenAI GPT-4o

Tạo nội dung tự nhiên, không robot

8 dự án được theo dõi:

infinitlabs, anoma, memex, uxlink, mitosis, virtuals, pharos, zama

⚙️ Cấu hình
Thêm API keys vào file .env:

env
TWITTER_API_KEY=your_key
TWITTER_API_SECRET=your_secret
TWITTER_ACCESS_TOKEN=your_token
TWITTER_ACCESS_SECRET=your_secret
TWITTER_BEARER_TOKEN=your_bearer_token
GEMINI_API_KEY=your_gemini_key  # Ưu tiên
OPENAI_API_KEY=your_openai_key  # Dự phòng
🛠️ Sử dụng
bash
# Chế độ chính
python bot.py

# Lệnh kiểm tra:
python bot.py test       # Test tạo tweet
python bot.py quote      # Test tweet trích dẫn
python bot.py time       # Test giọng điệu theo thời gian
python bot.py analytics  # Kiểm tra analytics
python bot.py mentions   # Kiểm tra mentions
🌟 Đặc điểm nổi bật
Smart Opening: Mở bài thông minh dựa trên lịch sử

python
# Ví dụ smart openings
smart_openings = {
    "first_discovery": ["saw {clean_project_name} the other day..."],
    "recent_follow_up": ["took another look at {clean_project_name}..."],
    # ... 6 loại khác
}
Time-Based Tone: Giọng điệu thay đổi theo thời gian

python
# 6AM-12PM: energetic_morning
# 12PM-5PM: analytical_noon
# 5PM-10PM: casual_evening
# 10PM-6AM: chill_night
Đa dạng nội dung:

Phân tích kỹ thuật (tech_deep_dive)

So sánh dự án (comparison)

Dự đoán tương lai (future_prediction)

Giải thích bằng ẩn dụ (daily_metaphor)

📊 Quản lý dữ liệu
tweet_history.json: Lưu lịch sử bài đăng

tweet_analytics.json: Theo dõi hiệu suất tweet

replied_tweets.json: Lưu trữ các tweet đã trả lời

⚠️ Lưu ý
Bot chỉ hoạt động từ 8:00 - 24:00 (giờ EU)

Tự động giới hạn 12 tweet/ngày (mỗi 1.33h)

Ưu tiên sử dụng Gemini API cho chất lượng tốt nhất