from dataclasses import dataclass

@dataclass
class VizWizConfig:
    PROMPT_TEMPLATE: str = '''You are an advanced video question-answering AI assistant. You have been provided with an image and a question related to the image. Your task is to carefully analyze the image and provide the answer to the question. The question can be unanswerable, if there is no appropriate information on image. Your answer can be yes, no, a number, unanswerable or any other.

    Question: {}
    '''
    PROMPT_UNCERTANTY: str = '''You are an advanced video question-answering AI assistant. You have been provided with an image and a question related to the image. Your task is to carefully analyze the image and provide the answer to the question.

    Question: {}
    '''
    dataset_dir: str = "/path/to/VizWiz"