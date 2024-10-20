import random
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime

def generate_addition_problem(existing_problems):
    while True:
        a = random.randint(1, 19)
        b = random.randint(1, 19)
        problem = f"{a} + {b} = "
        if a + b <= 20 and problem not in existing_problems:
            existing_problems.add(problem)
            return problem

def generate_problems(rows, cols):
    problems = []
    existing_problems = set()
    for _ in range(rows):
        row = [generate_addition_problem(existing_problems) for _ in range(cols)]
        problems.append(row)
    return problems

def print_problems_to_pdf(problems, filename):
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4
    margin = 50
    line_height = (height - 2 * margin) / 10  # 10 lines per page
    section_width = (width - 2 * margin) / 3
    y = height - margin

    # Set font for problems
    c.setFont("Helvetica", 15)

    for row in problems:
        for i, text in enumerate(row):
            x = margin + i * section_width
            c.drawString(x, y, text)
        y -= line_height
        if y < margin:
            c.showPage()
            y = height - margin

    c.save()

if __name__ == "__main__":
    rows, cols = 10, 3
    problems = generate_problems(rows, cols)
    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"math_problems_{current_time}.pdf"
    print_problems_to_pdf(problems, filename)