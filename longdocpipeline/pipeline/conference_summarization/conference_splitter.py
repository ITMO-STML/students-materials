from typing import List

from longdocpipeline.pipeline.conference_summarization.input_text_preprocessor import InputTxtPreprocessor
from longdocpipeline.pipeline.conference_summarization.diarization_corrector import DiarizationCorrector
from longdocpipeline.pipeline.conference_summarization.talks_splitter import TalksIdentifierSplitter

from langchain_community.chat_models import GigaChat

class ConferenceSplitter:
    """
    Splits conference text into JSON chunks,
    corrects diarization mistakes,
    identifies type of conference parts and splits it into talks / questions sections
    """
    def __init__(
        self,
        llm : GigaChat,
        system_prompt : str,
        user_prompt : str  
    ):
        self.llm = llm
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

    def preprocess_text(
            self,
            path_to_txt : str
    ) -> List[dict]:
        """
        Reads text from .txt file and splits it into JSON chunks
        """
        input_preprocessor = InputTxtPreprocessor(self.llm)
        splitted_chunks = input_preprocessor.split_text_into_chunks(path_to_txt)
        return splitted_chunks

    def correct_diarization(
            self,
            splitted_chunks : List[dict]
    ) -> List[dict]:
        """
        Corrects diarization mistakes
        """
        diar_corrector = DiarizationCorrector()
        chunks_correct_diar = []
        for chunk in splitted_chunks:
            chunk_corrected_diar = diar_corrector.correct_diar_mistakes(chunk)
            chunks_correct_diar.append(chunk_corrected_diar)
        return chunks_correct_diar
    
    def identify_conference_type_boundaries(
            self,
            chunks_correct_diar : List[dict]
    ) -> List[dict]:
        """
        Identifies type of conference part and splits it into talks / questions sections
        """
        talks_splitter = TalksIdentifierSplitter(
            self.llm,
            self.system_prompt,
            self.user_prompt
            )
        conference_talks = talks_splitter.get_conference_talks(chunks_correct_diar)
        return conference_talks

    def split_conference_into_talks(
            self,
            path_to_txt : str
    ) -> List[dict]:
        """
        Splits conference text into JSON chunks,
        corrects diarization mistakes,
        identifies type of conference parts and splits it into talks / questions sections
        """
        splitted_chunks = self.preprocess_text(path_to_txt)
        chunks_correct_diar = self.correct_diarization(splitted_chunks)
        conference_talks = self.identify_conference_type_boundaries(chunks_correct_diar)
        return conference_talks
