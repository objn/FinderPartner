# 🤖 AI Profile Matcher

โครงการ Python ที่ใช้ CLIP (Contrastive Language-Image Pre-training) ในการประเมินว่าโปรไฟล์ (รูปภาพหลายภาพ) ตรงกับคำบรรยายข้อความหรือไม่ เพื่อตัดสินใจ LIKE หรือ UNLIKE

## ✨ คุณสมบัติหลัก

- 🧠 ใช้โมเดล CLIP ล่าสุดสำหรับการเปรียบเทียบข้อความและรูปภาพ
- ⚡ รองรับทั้ง GPU (CUDA) และ CPU (auto-detect)
- 📊 หลายวิธีในการรวมคะแนนจากหลายภาพ (mean, max, weighted_mean, top_k)
- 🎯 ปรับแต่งค่า threshold ได้ตามต้องการ
- 🔧 กำหนดค่าผ่าน environment variables หรือ command line
- 📱 รองรับ Unicode path (Windows/Unix)
- 🗂️ ประมวลผลรูปภาพแบบ batch เพื่อประสิทธิภาพ

## 🔧 การติดตั้ง

### 1. สร้าง Virtual Environment

```bash
# สร้าง virtual environment
python -m venv .venv

# เปิดใช้งาน (Windows)
.venv\Scripts\activate

# เปิดใช้งาน (Linux/Mac)
source .venv/bin/activate
```

### 2. ติดตั้งไลบรารี

```bash
pip install -r requirements.txt
```

### 3. ตั้งค่า Environment (ไม่บังคับ)

```bash
# คัดลอกไฟล์ตัวอย่าง
copy .env.example .env

# แก้ไขค่าต่างๆ ในไฟล์ .env
```

## 🚀 วิธีใช้งาน

### การใช้งานพื้นฐาน

```bash
# ใช้โฟลเดอร์ default (./temp)
python src/main.py --prompt "สาวเอเชีย ใส่แว่น ผมยาว"

# ระบุโฟลเดอร์รูปภาพเอง
python src/main.py --prompt "cute girl with glasses" --img_dir ./profiles/user123

# ปรับ threshold
python src/main.py --prompt "สาวสวยใส" --threshold 0.3
```

### โหมด Interactive

```bash
python src/main.py --interactive --img_dir ./temp
```

### ตัวเลือกการรวมคะแนน

```bash
# ใช้คะแนนสูงสุดแทนค่าเฉลี่ย
python src/main.py --prompt "สาวน่ารัก" --method max

# ใช้ weighted average
python src/main.py --prompt "สาวน่ารัก" --method weighted_mean

# ใช้ค่าเฉลี่ยของ top 3 คะแนน
python src/main.py --prompt "สาวน่ารัก" --method top_k
```

### ตัวเลือกการแสดงผล

```bash
# แสดงคะแนนทุกภาพ
python src/main.py --prompt "สาวน่ารัก" --show_all

# โหมดเงียบ (แสดงแค่ผลตัดสิน)
python src/main.py --prompt "สาวน่ารัก" --quiet

# โหมด verbose (แสดงรายละเอียดเพิ่ม)
python src/main.py --prompt "สาวน่ารัก" --verbose
```

## 📁 โครงสร้างโปรเจค

```
ai_profile_matcher/
├── src/
│   ├── __init__.py          # Package initialization
│   ├── main.py              # CLI entry point
│   ├── embedding.py         # CLIP text/image encoding
│   ├── scorer.py            # Scoring and decision logic
│   └── utils.py             # Utility functions
├── temp/                    # รูปภาพตัวอย่าง (ผู้ใช้เตรียมเอง)
├── requirements.txt         # Python dependencies
├── .env.example            # ไฟล์ตั้งค่าตัวอย่าง
└── README.md               # คู่มือนี้
```

## ⚙️ การตั้งค่า

### Environment Variables (.env)

```env
# CLIP Model Configuration
MODEL_NAME=ViT-L-14        # โมเดล CLIP ที่ใช้
PRETRAINED=openai          # Pretrained weights

# Scoring Configuration  
THRESHOLD=0.25             # เกณฑ์ตัดสิน LIKE/UNLIKE
SCORE_METHOD=mean          # วิธีรวมคะแนน

# Processing Configuration
BATCH_SIZE=32              # ขนาด batch สำหรับประมวลผล

# Logging Configuration
LOG_LEVEL=INFO             # ระดับ logging
```

### วิธีการรวมคะแนน (Score Methods)

- **mean**: ค่าเฉลี่ยของคะแนนทุกภาพ (แนะนำ)
- **max**: คะแนนสูงสุดจากภาพใดภาพหนึ่ง
- **weighted_mean**: ค่าเฉลี่ยถ่วงน้ำหนัก (ให้น้ำหนักมากกับคะแนนสูง)
- **top_k**: ค่าเฉลี่ยของ top 3 คะแนน

### คำแนะนำค่า Threshold

- **0.15-0.20**: ผ่อนปรน (ยอมรับได้ง่าย)
- **0.25-0.30**: สมดุล (แนะนำ)
- **0.35-0.40**: เข้มงวด (เลือกเฉพาะที่ตรงมาก)
- **0.45+**: เข้มงวดมาก (เฉพาะที่ตรงที่สุด)

## 🖼️ การเตรียมรูปภาพ

### รูปแบบที่รองรับ
- JPG/JPEG
- PNG  
- BMP
- GIF
- TIFF
- WebP

### โครงสร้างโฟลเดอร์แนะนำ
```
profiles/
├── user001/          # โปรไฟล์คนที่ 1
│   ├── photo1.jpg
│   ├── photo2.jpg
│   └── photo3.jpg
├── user002/          # โปรไฟล์คนที่ 2
│   ├── img1.png
│   └── img2.png
└── temp/             # โฟลเดอร์ทดสอบ
    ├── pic1.jpg
    └── pic2.jpg
```

## 📊 ตัวอย่างผลลัพธ์

```
🎯 PROFILE EVALUATION RESULTS
============================================================

👍 DECISION: LIKE (confidence: HIGH)
📊 Profile Score: 0.3247
🎚️  Threshold: 0.2500
📈 Method: mean

📋 STATISTICS:
   Images processed: 5
   Average similarity: 0.3247
   Best match: 0.4892
   Worst match: 0.1823
   Standard deviation: 0.1156

🖼️  TOP 5 IMAGE SCORES:
--------------------------------------------------
⭐ photo1.jpg                    0.4892
⭐ photo3.jpg                    0.3841
⭐ photo2.jpg                    0.3245
📷 photo5.jpg                    0.2156
📷 photo4.jpg                    0.1823
```

## 🔧 การพัฒนาและปรับแต่ง

### การติดตั้งสำหรับพัฒนา

```bash
# ติดตั้ง development dependencies (ไม่บังคับ)
pip install black isort flake8

# Format โค้ด
black src/
isort src/

# ตรวจสอบ style
flake8 src/
```

### การเพิ่มโมเดล CLIP อื่นๆ

แก้ไขใน `.env`:
```env
MODEL_NAME=ViT-B-32    # หรือ ViT-L-14, RN50, RN101, etc.
PRETRAINED=openai      # หรือ laion2b_s34b_b79k, etc.
```

## 🐛 การแก้ไขปัญหา

### ปัญหาที่พบบ่อย

1. **CUDA Out of Memory**
   ```env
   BATCH_SIZE=16    # ลดขนาด batch
   ```

2. **โมเดลโหลดช้า**
   - โมเดลจะถูกดาวน์โหลดครั้งแรกและจัดเก็บไว้
   - การใช้งานครั้งต่อไปจะเร็วขึ้น

3. **รูปภาพโหลดไม่ได้**
   - ตรวจสอบไฟล์รูปภาพไม่เสียหาย
   - ตรวจสอบรูปแบบไฟล์ที่รองรับ

### การดู Log รายละเอียด

```bash
python src/main.py --prompt "test" --verbose
```

## 📝 TODO / การพัฒนาต่อ

- [ ] รองรับ video files
- [ ] Web interface
- [ ] Database integration
- [ ] Multi-language prompts optimization
- [ ] Performance benchmarking
- [ ] Docker containerization

## 🤝 การมีส่วนร่วม

1. Fork โปรเจค
2. สร้าง feature branch
3. Commit การเปลี่ยนแปลง
4. Push ไปยัง branch
5. สร้าง Pull Request

## 📄 License

โปรเจคนี้เป็น open source สำหรับการศึกษาและใช้งานส่วนตัว

---

Made with ❤️ by AI Developer
