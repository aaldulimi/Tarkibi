import requests
import json
import subprocess
from pytube import YouTube
from . import general

def youtube_search(query: str):
    """
    Searches youtube for a given query and returns a list of videos
    """
    query = query.replace(' ', '+')
    url = f'https://www.youtube.com/results?search_query={query}'

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

    response = requests.get(url, headers=headers)

    start = 'var ytInitialData = '
    end = ';</script>'
    json_data = response.text.split(start)[1].split(end)[0]
    data = json.loads(json_data)
    
    videos = data['contents']['twoColumnSearchResultsRenderer']['primaryContents']['sectionListRenderer']['contents'][0]['itemSectionRenderer']['contents'][1:]

    valid_videos = []
    for video in videos:
      if 'videoRenderer' in video:
        valid_videos.append(video['videoRenderer'])
    
    results = []
    for video in valid_videos:
      title = video['title']['runs'][0]['text']
      length = video['lengthText']['simpleText']
      video_id = video['videoId']

      results.append({
        'title': title,
        'length': length,
        'id': video_id,
        'url': f'https://www.youtube.com/watch?v={video_id}'
       })
      
    return results  

def download_video(video_id: str, output_path: str = f'{general.BASE_DIR}/{general.AUDIO_RAW}') -> None:
    url = f'https://www.youtube.com/watch?v={video_id}'
    
    yt = YouTube(url)
    audio_file = yt.streams.filter(only_audio=True).get_audio_only()

    file_name = video_id + '.mp4'
    download_dir = f'{general.BASE_DIR}/{general.DOWNLOADS}'
    audio_file.download(output_path=download_dir, filename=file_name)

    subprocess.run(f'ffmpeg -i "{download_dir}/{file_name}" -ac 2 -f wav {output_path}/{video_id}.wav', shell=True)
