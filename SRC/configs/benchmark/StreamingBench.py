from dataclasses import dataclass

@dataclass
class StreamingBenchConfig:
    PROMPT_TEMPLATE: str = '''You are an advanced video question-answering AI assistant. You have been provided with some frames from the video and a multiple-choice question related to the video. Your task is to carefully analyze the video and provide the best answer to question, choosing from the four options provided. Respond with only the letter (A, B, C, or D) of the correct option.

    Question: {}

    Options:
    {}
    {}
    {}
    {}'''

    PROMPT_TEMPLATE_WITHOUT_OPTIONS: str = '''You are an advanced video question-answering AI assistant. You have been provided with a video and a question related to the video. Your task is to carefully analyze the video and provide the answer to the question. 

    Question: {}
    '''
    dataset_dir: str = "/path/to/StreamingBench"