import requests
import json
import subprocess
from pytube import YouTube
from . import general
from tarkibi.utilities._config import logger
import os

logger = logger.getChild(__name__)


class _Youtube:
    # should make this random
    _HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    _DOWNLOADS_OUTPUT_PATH = f"{general.BASE_DIR}/downloads"

    def __init__(self):
        pass

    def _search(self, query: str) -> list[dict[str, str]]:
        """
        Searches youtube for a given query and returns a list of videos
        parameters
        ----------
        query: str
            The query to search for

        returns
        -------
        list[dict[str, str]]
            A list of videos
        """
        logger.info(f"Tarkibi _search: Searching youtube for query: {query}")
        query = query.replace(" ", "+")
        url = f"https://www.youtube.com/results?search_query={query}"

        response = requests.get(url, headers=self._HEADERS)

        start = "var ytInitialData = "
        end = ";</script>"
        json_data = response.text.split(start)[1].split(end)[0]
        data = json.loads(json_data)

        videos = data["contents"]["twoColumnSearchResultsRenderer"]["primaryContents"][
            "sectionListRenderer"
        ]["contents"][0]["itemSectionRenderer"]["contents"][1:]

        valid_videos = []
        for video in videos:
            if "videoRenderer" in video:
                valid_videos.append(video["videoRenderer"])

        results = []
        for video in valid_videos:
            title = video["title"]["runs"][0]["text"]
            length = video["lengthText"]["simpleText"]
            video_id = video["videoId"]

            results.append(
                {
                    "title": title,
                    "length": length,
                    "id": video_id,
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                }
            )

        if not results:
            raise ValueError("No videos found. Try again.")

        return results

    # Pytube currently has error with downloading videos
    def _download_video(self, video_id: str, output_file_path: str) -> None:
        """
        Downloads a video from youtube and converts it to a wav file
        parameters
        ----------
        video_id: str
            The id of the video to download
        output_path: str
            The path to save the video to

        returns
        -------
        None
        """
        os.makedirs(self._DOWNLOADS_OUTPUT_PATH, exist_ok=True)
        logger.info(f"Tarkibi _download_video: Downloading video: {video_id}")
        url = f"https://www.youtube.com/watch?v={video_id}"

        yt = YouTube(url)
        audio_file = yt.streams.filter(only_audio=True).get_audio_only()

        file_name = video_id + ".mp4"
        audio_file.download(output_path=self._DOWNLOADS_OUTPUT_PATH, filename=file_name)

        subprocess.run(
            f'ffmpeg -i "{self._DOWNLOADS_OUTPUT_PATH}/{file_name}" -ac 2 -f wav {output_file_path}',
            shell=True,
        )

        os.remove(f"{self._DOWNLOADS_OUTPUT_PATH}/{file_name}") 

    def _download_video_dlc(self, video_id: str, output_dir: str) -> None:
        """
        Downloads a video from youtube and converts it to a wav file
        parameters
        ----------
        video_id: str
            The id of the video to download
        output_path: str
            The path to save the video to

        returns
        -------
        None
        """
        url = f"https://www.youtube.com/watch?v={video_id}"
        youtube_dl_cmd = f'cd {output_dir} && youtube-dlc --extract-audio --audio-format wav --output "%(id)s.%(ext)s" {url}',
        subprocess.run(youtube_dl_cmd, shell=True, check=True)
