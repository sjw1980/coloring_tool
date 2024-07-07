import requests
from PIL import Image
import os
from bs4 import BeautifulSoup
import re
from datetime import datetime
from pdf2image import convert_from_path
import cv2
import pytesseract
import numpy as np

def download_image(url, filename):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(filename, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)

def download_one_image(url, ext):
    # 현재 파일의 디렉토리를 가져온다.
    current_dir = os.path.dirname(os.path.realpath(__file__))
    download_dir = os.path.join(current_dir, "download")

    # "download" 폴더가 없으면 생성한다.
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    # 주어진 URL의 웹 페이지를 가져온다.
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")

    # 웹 페이지에서 모든 링크를 찾는다.
    links = soup.find_all('a', href=True)

    # 링크 중에서 .pdf 파일을 찾는다.
    for link in links:
        href = link['href']
        if href.endswith(ext):
            # 파일 이름을 URL에서 추출한다.
            filename = re.search(r'/([\w_-]+[.][\w\d_-]+)$', href, re.IGNORECASE)
            if filename:
                filename = filename.group(1)
                # 파일을 다운로드하고 "download" 폴더에 저장한다.
                response = requests.get(href, stream=True)
                if response.status_code == 200:
                    filepath = os.path.join(download_dir, filename)
                    with open(filepath, 'wb') as file:
                        for chunk in response.iter_content(1024):
                            file.write(chunk)
    

def download_all_qr_images():
    # loop 1 to 21
    for i in range(1, 22):
        url = f"https://i0.wp.com/star-shape.com/wp-content/uploads/2023/11/{i}.png"
        filename = f"downloaded_qr_{i}.png"
        download_image(url, filename)

def make_one_qr_image():
    Image.MAX_IMAGE_PIXELS = None
    # image_dir is current file directory
    image_dir = os.path.dirname(os.path.realpath(__file__))

    # Margin in pixels
    margin = 50

    # A4 size at 300dpi
    a4_width = 2480
    a4_height = 3508

    # Calculate new image size
    new_image_width = a4_width - 2 * margin
    new_image_height = a4_height - 2 * margin

    # Calculate size for each image
    image_width = new_image_width // 3
    image_height = new_image_height // 7

    # Open and resize images
    images = []
    for image in os.listdir(image_dir):
        if image.endswith('.png'):
            img = Image.open(os.path.join(image_dir, image))
            img.thumbnail((image_width, image_height))
            images.append(img)

    # Create new image with white background
    new_image = Image.new('RGB', (a4_width, a4_height), (255, 255, 255))

    # Paste images into new image
    for index, image in enumerate(images):
        row = index // 3
        col = index % 3
        new_image.paste(image, (margin + col * image_width, margin + row * image_height))

    # Save new image
    new_image.save('combined_image.png')

def pdf_to_png(pdf_path):
    # PDF 파일의 첫 페이지만 이미지로 변환
    images = convert_from_path(pdf_path, first_page=1, last_page=1)
    
    if images:
        # 변환된 이미지를 가져온다 (첫 페이지만 변환했으므로 첫 번째 이미지만 사용)
        image = images[0]
        
        # PDF 파일 이름에서 확장자를 제외한 부분을 가져온다
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        # 동일한 디렉토리에 PNG 파일로 저장
        save_path = os.path.join(os.path.dirname(pdf_path), f"{base_name}.png")
        image.save(save_path, 'PNG')
        
        print(f"Saved: {save_path}")
    else:
        print("No images were created from the PDF.")


def make_one_image(path, ext):
    # 경로에서 모든 파일을 찾는다.
    files = [f for f in os.listdir(path) if f.endswith(ext)]
    
    images = []  # 이미지 객체를 저장할 리스트

    for file in files:
        full_path = os.path.join(path, file)
        
        try:
            cv2_image = cv2.imread(full_path)

            cv2_image = remove_texts(cv2_image)
            cv2_image = remove_margin(cv2_image)

            # BGR에서 RGB로 색상 순서를 변환합니다.
            cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

            pil_image = Image.fromarray(cv2_image)
            images.append(pil_image)

        except IOError:
            print(f"Cannot open {file}")

    # 6개의 이미지를 하나의 큰 이미지로 결합한다.
    # A4 용지 크기를 픽셀 단위로 설정 (300 DPI 기준)
    a4_width, a4_height = 2480, 3508
    # 각 이미지를 1/6 크기로 조정
    target_width, target_height = a4_width // 2, a4_height // 3
    for i in range(0, len(images), 6):
        group = images[i:i+6]   
        new_im = Image.new('RGB', (a4_width, a4_height), (255, 255, 255))

        for j, img in enumerate(group):
            # 원본 이미지의 비율을 유지하면서 A4 용지의 한 변에 맞게 조정
            img_ratio = img.width / img.height
            target_ratio = target_width / target_height

            if img_ratio > target_ratio:
                # 이미지의 너비가 타겟 비율보다 클 경우, 너비를 기준으로 높이 조정
                resize_width = target_width
                resize_height = round(target_width / img_ratio)
            else:
                # 이미지의 높이가 타겟 비율보다 클 경우, 높이를 기준으로 너비 조정
                resize_height = target_height
                resize_width = round(target_height * img_ratio)

            img_resized = img.resize((resize_width, resize_height), Image.Resampling.LANCZOS)
            # 조정된 이미지를 새 이미지에 붙임. 이미지가 중앙에 오도록 계산
            x_offset = (target_width - resize_width) // 2 + (j % 2) * target_width
            y_offset = (target_height - resize_height) // 2 + (j // 2) * target_height
            new_im.paste(img_resized, (x_offset, y_offset))

        # 현재 시간을 기반으로 한 고유한 파일 이름 생성
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'output_image_{i//6}_{timestamp}.jpg'
        
        # 결합된 이미지를 저장
        new_im.save(os.path.join(path, filename))

def convert_all_pdf(path):
    # 경로에서 모든 파일을 찾는다.
    files = [f for f in os.listdir(path) if f.endswith('.pdf')]
    
    for file in files:
        full_path = os.path.join(path, file)
        pdf_to_png(full_path)

def pdf_to_png(pdf_path):
    # PDF 파일의 첫 페이지만 이미지로 변환
    images = convert_from_path(pdf_path, first_page=1, last_page=1)
    
    if images:
        # 변환된 이미지를 가져온다 (첫 페이지만 변환했으므로 첫 번째 이미지만 사용)
        image = images[0]
        
        # PDF 파일 이름에서 확장자를 제외한 부분을 가져온다
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        # 동일한 디렉토리에 PNG 파일로 저장
        save_path = os.path.join(os.path.dirname(pdf_path), f"{base_name}.png")
        image.save(save_path, 'PNG')
        
        print(f"Saved: {save_path}")
    else:
        print("No images were created from the PDF.")


def make_image_to_pdf(path, ext):
    # 지정된 경로에서 주어진 확장자를 가진 모든 파일 찾기
    images = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(ext)]
    
    # 이미지 파일을 열고, 변환을 위한 리스트 생성
    image_list = []
    for image in images:
        img = Image.open(image)
        img_converted = img.convert('RGB')
        image_list.append(img_converted)
    
    # 이미지 리스트가 비어있지 않은 경우 PDF 생성
    if image_list:
        # 첫 번째 이미지를 기준으로 PDF 생성, 나머지 이미지 추가
        pdf_path = os.path.join(path, "output.pdf")
        image_list[0].save(pdf_path, save_all=True, append_images=image_list[1:])
        print(f"PDF has been created at {pdf_path}")
    else:
        print("No images found with the given extension.")

# get all images from the website of wordpress
def fetch_all_media(url, search = None):
    page = 1
    total_pages = 1  # 시작값은 1로 설정, 실제 페이지 수는 첫 번째 요청 후에 알 수 있음
    source_urls = []

    while page <= total_pages:
        # 현재 페이지 번호를 URL에 추가

        response = None
        if search is None:
            response = requests.get(url, params={'page': page, 'per_page': 100})
        else:
            response = requests.get(url, params={'page': page, 'per_page': 100, 'search': search})

        if response.status_code == 200:
            # 첫 요청에서 전체 페이지 수를 확인
            if page == 1:
                total_pages = int(response.headers['X-WP-TotalPages'])

            # JSON 데이터를 파이썬 객체로 변환
            media_items = response.json()

            # 이미지 목록 출력
            for item in media_items:
                print(item['source_url'])  # 이미지 URL 출력
                source_urls.append(item['source_url'])

            page += 1  # 다음 페이지로
        else:
            print(f'Failed to retrieve media items on page {page}')
            break
    with open('source_urls.txt', 'w') as file:
        for url in source_urls:
            file.write(url + '\n')

def download_files_from_list(file_list_path, download_folder='downloads'):
    # 다운로드 폴더가 없으면 생성
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)
    
    with open(file_list_path, 'r') as file:
        for line in file:
            url = line.strip()
            if url:
                try:
                    response = requests.get(url)
                    if response.status_code == 200:
                        # URL에서 파일 이름 추출
                        file_name = url.split('/')[-1]
                        file_path = os.path.join(download_folder, file_name)
                        
                        # 파일 저장
                        with open(file_path, 'wb') as f:
                            f.write(response.content)
                        
                        # 파일이 이미지인 경우 PNG 형식으로 변환
                        if file_name.lower().endswith(('.jpg', '.jpeg', '.gif', '.bmp')):
                            # 원본 이미지 열기
                            img = Image.open(file_path)
                            # PNG 형식으로 변환된 파일 경로
                            png_file_path = os.path.splitext(file_path)[0] + '.png'
                            # PNG로 저장
                            img.save(png_file_path)
                            # 원본 파일 삭제
                            os.remove(file_path)
                            print(f"Downloaded and converted {file_name} to PNG format in {download_folder}")
                        else:
                            print(f"Downloaded {file_name} to {download_folder}")

                    else:
                        print(f"Failed to download {url}")
                except Exception as e:
                    print(f"Error downloading {url}: {e}")

def remove_texts(cv_image, show_image = False):
    # 이미지 로드
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,3))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,3))
    dilate = cv2.dilate(close, dilate_kernel, iterations=1)

    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 800 and area < 15000:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(cv_image, (x, y), (x + w, y + h), (255,255,255), -1)
    
    if(show_image):
        #결과 이미지 표시
        cv2.imshow('Image with Text Boxes', cv_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return cv_image

def remove_margin(cv_image, show_image = False):
    gray = 255*(cv_image < 128).astype(np.uint8) # 이미지 반전 및 텍스트를 흰색으로 만듦

    if len(gray.shape) > 2 and gray.shape[2] == 3: # BGR 이미지인 경우
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    elif len(gray.shape) > 2 and gray.shape[2] == 1: # 단일 채널이지만 추가 차원이 있는 경우
        gray = gray[:, :, 0]
        
    coords = cv2.findNonZero(gray) # 모든 비-제로(텍스트가 있는) 포인트 찾기
    x, y, w, h = cv2.boundingRect(coords) # 최소 경계 사각형 찾기
    rect = cv_image[y:y+h, x:x+w] # 원본 이미지에서 이미지 자르기

    if(show_image):
        #결과 이미지 표시
        cv2.imshow('Image with Text Boxes', rect)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return rect

def auto_process(url, search):
    fetch_all_media(url, search)
    download_files_from_list('source_urls.txt', search)
    make_one_image(search, '.png')
    make_image_to_pdf(search, '.jpg')

def is_grayscale_or_bw(image):    
    # Check if the image has only one channel
    if len(image.shape) == 2:
        # The image is grayscale
        return True
    elif len(image.shape) == 3 and image.shape[2] == 1:
        # The image is grayscale (with a single channel in the third dimension)
        return True
    elif len(image.shape) == 3:
        # Additional check for black and white
        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Check if all pixels are either 0 or 255
        unique_values = np.unique(gray_image)
        if np.all(np.isin(unique_values, [0, 255])):
            # The image is black and white
            return True
    # The image is not grayscale or black and white
    return False

def test_code(path):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,3))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,3))
    dilate = cv2.dilate(close, dilate_kernel, iterations=1)

    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 800 and area < 15000:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(image, (x, y), (x + w, y + h), (222,228,251), -1)

    cv2.imshow('image', image)
    cv2.waitKey()
    
#download_one_image('https://mondaymandala.com/hello-kitty-coloring-pages/', '.pdf')
#download_one_image('https://mondaymandala.com/keroppi-coloring-pages/', '.pdf')
#download_one_image('https://mondaymandala.com/my-melody-coloring-pages/', '.pdf')
#download_one_image('https://mondaymandala.com/kuromi-coloring-pages/', '.pdf')
#download_one_image('https://mondaymandala.com/sanrio-coloring-pages/', '.pdf')

#make_one_image('download', '.pdf')
#convert_all_pdf('download')
#make_one_image('download', '.png')
#make_image_to_pdf("download", ".jpg")
#make_one_image('download', '.png')
#make_image_to_pdf("download", ".jpg")

#fetch_all_media('https://www.just-coloring-pages.com/wp-json/wp/v2/media', 'pochacco')
#download_files_from_list('source_urls.txt', 'pochacco')
#make_one_image('pochacco', '.png')
#make_image_to_pdf('pochacco', '.jpg')

#test=remove_texts(cv2.imread('adorable-kuromi.png'), True)
#remove_margin(test, True)

auto_process('https://www.just-coloring-pages.com/wp-json/wp/v2/media', 'kuromi')

