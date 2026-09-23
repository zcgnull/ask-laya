"""生成 ModelScope 创空间封面图（webapp/studio/cover.png，1640x800，同 820x400 比例）。"""
from PIL import Image, ImageDraw, ImageFont

W, H = 1640, 800

CREAM = "#faf1dc"
PAPER = "#fffdf4"
INK = "#16130e"
RED = "#d93a2b"
GOLD = "#f0b429"
GREEN = "#2e7d46"
GRAY = "#6b6255"

img = Image.new("RGB", (W, H), CREAM)
d = ImageDraw.Draw(img)

# 点阵纹理
for x in range(18, W, 36):
    for y in range(18, H, 36):
        d.ellipse([x, y, x + 2, y + 2], fill="#e6d9bc")

F = "/System/Library/Fonts/Supplemental/"
CJK = "/System/Library/Fonts/Hiragino Sans GB.ttc"
georgia = lambda s: ImageFont.truetype(F + "Georgia Bold.ttf", s)
cjk = lambda s: ImageFont.truetype(CJK, s, index=0)
mono = lambda s: ImageFont.truetype(F + "Courier New Bold.ttf", s)


def hard_rect(x0, y0, x1, y1, fill, radius=14, shadow=8, border=3):
    d.rounded_rectangle([x0 + shadow, y0 + shadow, x1 + shadow, y1 + shadow],
                        radius=radius, fill=INK)
    d.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=fill, outline=INK, width=border)


def text_center(x, y, s, f, fill):
    b = d.textbbox((0, 0), s, font=f)
    d.text((x - (b[2] - b[0]) / 2, y - (b[3] - b[1]) / 2 - b[1]), s, fill=fill, font=f)


# ---------- 外框 ----------
d.rectangle([2, 2, W - 4, H - 4], outline=INK, width=6)
d.rectangle([16, 16, W - 18, H - 18], outline=INK, width=3)

# ---------- 左栏 ----------
hard_rect(64, 56, 124, 116, RED, radius=10, shadow=6, border=3)
text_center(94, 86, "L", georgia(40), PAPER)
d.text((142, 66), "问问 Laya", fill=INK, font=cjk(36))

d.text((64, 148), "Have a", fill=RED, font=georgia(84))
d.text((64, 240), "Question?", fill=RED, font=georgia(84))

d.text((66, 352), "专治日常纠结，单次前向给你裁决", fill=INK, font=cjk(27))
d.text((66, 396), "laya-multilingual · mmBERT-base", fill=GRAY, font=mono(21))
d.text((66, 424), "322M params · 0 output tokens", fill=GRAY, font=mono(21))

chips = ["该不该？", "选哪个？", "冲动指数"]
cx = 66
for i, c in enumerate(chips):
    w = d.textbbox((0, 0), c, font=cjk(20))[2] + 40
    active = i == 0
    hard_rect(cx, 486, cx + w, 538, INK if active else PAPER, radius=26, shadow=5, border=3)
    d.text((cx + 20, 500), c, fill=CREAM if active else INK, font=cjk(20))
    cx += w + 22

# ---------- 右栏：迷你裁决卡 ----------
cx0, cy0, cx1, cy1 = 830, 130, 1560, 560
hard_rect(cx0, cy0, cx1, cy1, PAPER, radius=18, shadow=12, border=4)
d.line([cx0 + 22, cy0 + 66, cx1 - 22, cy0 + 66], fill="#c9bda0", width=3)

d.text((cx0 + 22, cy0 + 20), "Laya 裁决书 · 签号 No.3538", fill=GRAY, font=cjk(18))
badge = "Best option (78%)"
bw = d.textbbox((0, 0), badge, font=georgia(18))[2] + 28
hard_rect(cx1 - 22 - bw, cy0 + 14, cx1 - 22, cy0 + 50, GREEN, radius=8, shadow=3, border=2)
d.text((cx1 - 22 - bw + 14, cy0 + 21), badge, fill=PAPER, font=georgia(18))

d.text((cx0 + 22, cy0 + 90), "Go with: in China!", fill=INK, font=georgia(38))

bars = [("in China", 0.775, GREEN), ("in the United States", 0.225, GOLD)]
by = cy0 + 165
bx0, bx1 = cx0 + 22, cx1 - 22
for label, v, color in bars:
    d.text((bx0, by), label, fill=INK, font=cjk(19))
    d.text((bx1 - 62, by + 1), f"{v*100:.1f}%", fill=INK, font=mono(17))
    d.rounded_rectangle([bx0, by + 34, bx1, by + 68], radius=8, fill="#ffffff", outline=INK, width=3)
    fw = int((bx1 - bx0) * v)
    d.rounded_rectangle([bx0 + 1, by + 35, bx0 + fw, by + 67], radius=7, fill=color)
    by += 108

d.text((bx0, cy1 - 48), "单次前向 106 ms · 0 token", fill=GRAY, font=cjk(18))

# ---------- 底部落款 ----------
d.text((66, H - 72), "modelscope.cn/studios/zcgnull/ask-laya", fill=GRAY, font=mono(20))

out = "/Volumes/zcgnull/project/laya/webapp/studio/cover.png"
img.save(out, optimize=True)
print("saved", out, img.size)
