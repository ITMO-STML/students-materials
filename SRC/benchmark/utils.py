import os
from pathlib import Path
from moviepy import VideoFileClip

def split_video(video_file: str | Path, start_time: int, end_time: int):
    """
    Split video into prefix part based on timestamp.
    video_file: path to video file
    start_time: start time in seconds
    end_time: end time in seconds
    """
    video_name = Path(video_file).stem
    output_dir = Path(Path(video_file).parent, "tmp_60")
    if not output_dir.exists():
        os.makedirs(output_dir)
    output_file = Path(output_dir, f"{video_name}_{start_time}_{end_time}.mp4")

    if output_file.exists():
        print(f"Video file {output_file} already exists.")
        return output_file

    if "sample_332" in video_file:
        return None

    video = VideoFileClip(video_file)
    video.subclipped(min(start_time, int(video.duration-1)), min(end_time, video.duration)).write_videofile(output_file, codec='libx264', audio=False)
    video.close()

    print(f"Video: {output_file} splitting completed.")
    return output_file


def convert_time(time : str) -> int:
    """Converts time like "00:03:10" to seconds"""
    return sum(int(x) * 60 ** i for i, x in enumerate(reversed(time.split(":"))))


def convert_path(path: str, dataset_dir: str) -> str:
    """Converts ./videos/sample_N_real.mp4 to `dataset_dir`/Real_Time_Visual_Understanding/sample_N/video.mp4

    Args:
    - path: str - orig path
    - dataset_dir: str - path to StreamingBench folder
    """
    folder = Path(path).stem[:-5]
    return dataset_dir + "/Real_Time_Visual_Understanding/" + folder + "/video.mp4"
