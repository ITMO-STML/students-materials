from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser

from longdocpipeline.pipeline.gigachat_provider import gigachat
from longdocpipeline.pipeline.conference_summarization.conference_splitter import ConferenceSplitter
from longdocpipeline.pipeline.inference_manager import LongDocInferenceManager, ProcessDocResult
from longdocpipeline.pipeline.conference_summarization.schemas import ConferenceTalkStructure, FinalSummaryStructure


class ConferenceSummarizer:
    """
    Splits conference to conference talks and/or questions sections
    and summarizes it using concat algorithm
    """
    def __init__(
            self,
            system_prompt_sum : str,
            user_prompt_sum : str
    ):
        self.system_prompt_sum = system_prompt_sum
        self.user_prompt_sum = user_prompt_sum

    def split_conference_into_talks(
            self,
            path_to_txt : str
    ) -> List[dict]:
        """
        Splits conference into conference talks or questions sections
        """
        conference_splitter = ConferenceSplitter(
            gigachat, 
            self.system_prompt_sum, 
            self.user_prompt_sum
            )
        conference_talks = conference_splitter.split_conference_into_talks(path_to_txt)
        return conference_talks
    
    def summarize_conference(
            self,
            path_to_txt : str
    ) -> ProcessDocResult:
        """
        Summarizes conference talks and/or questions sections
        """
        conference_talks = self.split_conference_into_talks(path_to_txt)
        
        talk_docs = []
        for talk in conference_talks:
            doc = Document(page_content=str(talk), metadata={})
            talk_docs.append(doc)

        task = "sum_conference"
        algorithm = "map_reduce"
        ldim = LongDocInferenceManager()

        map_parser = PydanticOutputParser(pydantic_object=ConferenceTalkStructure)
        iter_parser = PydanticOutputParser(pydantic_object=FinalSummaryStructure)

        oneshot = True if len(talk_docs) == 1 else False
        result = ldim.process_doc(talk_docs, task, algorithm, oneshot=oneshot, 
                                        map_parser=map_parser,
                                        iter_parser=iter_parser)
        return result.result_dict["final_result"]
