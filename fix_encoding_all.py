# fix_encoding_all.py
import chardet
from pathlib import Path

target = Path("train_phobert_sentiment_v2.py")  # đổi nếu bạn dùng file khác

raw = target.read_bytes()
enc = chardet.detect(raw)["encoding"]
print("🔍 Phát hiện encoding gốc:", enc)

text = raw.decode(enc or "utf-8", errors="ignore")

# Thay toàn bộ ký tự “lạ” bằng ASCII tương đương
replacements = {
    "–": "-", "—": "-", "―": "-", "−": "-",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "…": "...", "•": "-", " ": " "  # thay non-breaking space
}

for k, v in replacements.items():
    text = text.replace(k, v)

# Bổ sung dòng encoding đầu file nếu chưa có
if not text.startswith("# -*- coding: utf-8 -*-"):
    text = "# -*- coding: utf-8 -*-\n" + text

target.write_text(text, encoding="utf-8")
print("✅ Đã chuẩn hoá và lưu lại bằng UTF-8:", target)
