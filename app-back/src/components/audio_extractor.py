"""
Module to extract audio from a video file.
"""

from moviepy.video.io.VideoFileClip import VideoFileClip

def extract_audio_from_video(video_path, output_audio_path):
    """
    Extract audio from a video file and save to `output_audio_path`.

    Args:
        video_path: path to input video file
        output_audio_path: where the extracted audio will be saved
    """
    with VideoFileClip(video_path) as video:
        audio = video.audio
        if audio is not None:
            audio.write_audiofile(output_audio_path)
        else:
            raise ValueError("The video does not contain an audio track.")
