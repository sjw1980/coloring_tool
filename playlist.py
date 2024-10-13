import os
from pytube import YouTube
from pytube.download_helper import (
    download_videos_from_channels,
    download_video,
    download_videos_from_list,
)
from pydub import AudioSegment
import yt_dlp

def download_video_to_mp3(filename):
    output_path = './mp3'
    with open(filename, "r") as f:
        links = f.read().splitlines()
        for url in links:
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'outtmpl': output_path + '/%(playlist_title)s/%(playlist_index)s - %(title)s.%(ext)s',
                'yes_playlist': True,
                'ignoreerrors': True,  # Skip errors and continue downloading
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])



def download_video_list(filename):
    with open(filename, "r") as f:
        links = f.read().splitlines()
        for video in links:
#                download_video(url=video)
            try:
                download_video(url=video)
            except Exception as e:
                print("Exception: ", e)
                print("video: ", video)

def convert_videos_to_mp3(video_dir, output_dir):
    # 비디오 디렉토리의 모든 파일 목록 가져오기
    files = os.listdir(video_dir)
    
    # 출력 디렉토리가 없으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for file in files:
        # 파일 경로 설정
        video_path = os.path.join(video_dir, file)
        base, ext = os.path.splitext(file)
        mp3_path = os.path.join(output_dir, base + '.mp3')
        
        # 오디오 파일 변환
        audio = AudioSegment.from_file(video_path)
        audio.export(mp3_path, format="mp3")
        print(f"Converted {video_path} to {mp3_path}")

# download_video_list("playlist.txt")

# convert_videos_to_mp3("./videos", "./mp3")

download_video_to_mp3("playlist.txt")