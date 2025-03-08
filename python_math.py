import random
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab import rl_config

# 폰트 검색 경로 설정
rl_config.TTFSearchPath.append(r'C:\Users\Jinwoo\AppData\Local\Microsoft\Windows\Fonts')

# 한글 폰트 등록
pdfmetrics.registerFont(TTFont('ArialUnicodeMS', 'ARIALUNI.TTF'))

def create_star_problem(input_text, c, y, padding):
    stars = ['◇ ' for _ in range(9)]
    
    c.setFont('ArialUnicodeMS', 12)
    c.drawString(100, y, input_text)
    c.setFont('ArialUnicodeMS', 36)
    c.drawString(200, y - 10 , ''.join(stars))


count_data = ["하나", "둘", "셋", "넷", "다섯", "여섯", "일곱", "여덟", "아홉"]
order_data = ["첫", "두", "세", "네", "다섯", "여섯", "일곱", "여덟", "아홉"]

def make_problem(number, c, y):
    padding = 20 - len(count_data[number - 1]) * 3 - 3
    create_star_problem(f"{count_data[number - 1]}({number})", c, y, padding)
    y -= 20
    padding = 20 - len(order_data[number - 1]) * 3 - 7
    create_star_problem(f"{order_data[number - 1]}번째", c, y, padding)
    return y - 40

if __name__ == "__main__":
    c = canvas.Canvas("output.pdf", pagesize="A4")
    y = 750

    numbers = random.sample(range(1, 10), 9)
    
    for i in range(1, 8):
        y -= 10
        c.setFont('ArialUnicodeMS', 12)
        c.drawString(100, y, f"문제. {i}")
        y -= 20
        random_number = random_number = numbers.pop()
        y = make_problem(random_number, c, y)
        y -= 10
    
    c.save()
    print("PDF 파일이 생성되었습니다.")
