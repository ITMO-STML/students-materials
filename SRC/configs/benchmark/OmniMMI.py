from dataclasses import dataclass

@dataclass
class OmniMMIConfig:
    dataset_dir: str = "/path/to/OmniMMI"
    output_fps: float = 1.0
    YES_NO_TEMPLATE = """You are an advanced video question-answering AI assistant. You have been provided with some frames from the video and a question related to the video. Your task is to carefully analyze the video and provide the best answer to question, choosing from yes and no. Respond with only the yes or no.

    Question: {}"""
    MULTICHOICE_TEMPLATE: str = '''You are an advanced video question-answering AI assistant. You have been provided with some frames from the video and a multiple-choice question related to the video. Your task is to carefully analyze the video and provide the best answer to question, choosing from the four options provided. Respond with only the letter (A, B, C, or D) of the correct option.

    Question: {}

    Options:
    {}
    {}
    {}
    {}'''
    MULTICHOICE_PROACTIVE: str = '''You are an advanced video question-answering AI assistant. You have been provided with some frames from the video and a multiple-choice question related to the video. Your task is to carefully analyze provided frames and give the best answer to question, choosing from the four options provided. Response only with the letter (A, B, C, D or E) of the correct option, no other text needed.

    Question: {}

    Options:
    {}
    '''
    YES_NO_PROACTIVE = """You are an advanced video question-answering AI assistant. You have been provided with some frames from the video and a question related to the video. Your task is to carefully analyze the video and provide the best answer to question based on the current frame, choosing from yes and no. Respond with only the yes or no.

    Question: {}"""
    ALERTING_PROACTIVE = """ Print yes, if the event occures. Otherwise print \"not yet\""""