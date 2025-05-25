import re
import json
from langchain_community.chat_models import GigaChat
from langchain.text_splitter import CharacterTextSplitter
from typing import List
import ast
from tqdm import tqdm
from longdocpipeline.pipeline.constants import TASK_SPECIFIC_PREPROC_PARAMS


class InputTxtPreprocessor:
    """
    Converts conference text file to the list of chunks;
    every chunk contains list of jsons and does not exceed a specified chunk_size
    """
    def __init__(
            self, 
            llm : GigaChat
    ):
        self.llm = llm
        self.chunk_size = TASK_SPECIFIC_PREPROC_PARAMS["sum_conference"].chunk_size
        self.splitter = CharacterTextSplitter(
            separator="},",
            keep_separator="end",
            chunk_size=self.chunk_size,
            length_function=self._count_tokens,
        )

    @staticmethod
    def read_text(
            path_to_file : str
    ) -> List[str]:
        """
        Reads text file with conference transcript
        """
        with open(path_to_file, "r") as file:
            text_lines = file.read().splitlines()
        return text_lines

    @staticmethod
    def texts_to_json(
            text_lines : List[str]
    ) -> dict[List[dict]]:
        """
        Transforms text lines to json dicts
        """
        input_json = {"transcript": []}
        speaker_regex = re.compile(r"(.+)\s\[")
        starttime_regex = re.compile(r"\[(.+)\:")
        duration_regex = re.compile(r"\:(.+)\]")
        text_regex = re.compile(r"\]\s(.+)")
        for i, line in tqdm(enumerate(text_lines),
                            total=len(text_lines),
                            desc="Converting text lines to json chunks..."):
            speaker = re.findall(speaker_regex, line)[0]
            startTime = float(re.findall(starttime_regex, line)[0])
            endTime = round(startTime + float(re.findall(duration_regex, line)[0]), 2)
            text = re.findall(text_regex, line)[0]
            input_json["transcript"].append({
                "number": i,
                "startTime": startTime,
                "endTime": endTime,
                "speaker": int(speaker),
                "text": text
            })
        return ast.literal_eval(json.dumps(input_json, ensure_ascii=False))

    def _count_tokens(
            self, 
            string: str
    ) -> int:
        """
        Counts tokens in a string
        """
        return self.llm.get_num_tokens(string)
    
    def split_text_into_chunks(
            self,
            path_to_file : str
    ) -> List[dict]:
        """
        Creates chunks of a chunk_size;
        each chunk contains a list of jsons (as strings)
        """
        text_lines = self.read_text(path_to_file)
        input_jsons = self.texts_to_json(text_lines)
        input_jsons_str = str(input_jsons).replace("\'", "\"")
        start_regex = re.compile(r'{"transcript":\s\[')
        end_regex = re.compile(r'},$')
        json_chunks = []
        for chunk in self.splitter.split_text(input_jsons_str):
            if not re.match(start_regex, chunk):
                chunk = '{"transcript": [' + chunk
            if re.findall(end_regex, chunk):
                chunk = re.sub(end_regex, '}]}', chunk)
            json_chunks.append(eval(chunk))
        return json_chunks