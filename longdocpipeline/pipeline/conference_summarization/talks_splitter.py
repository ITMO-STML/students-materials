from typing import List
from tqdm import tqdm

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.chat_models import GigaChat

from longdocpipeline.pipeline.conference_summarization.schemas import TypeIdentifier


class TalksIdentifierSplitter:
    def __init__(
            self,
            llm : GigaChat,
            system_prompt : str,
            user_prompt : str
    ):
        self.chunks_with_corrected_diar = None
        self.llm = llm
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

    def identify_part_type(
            self,
            context : List[dict]
    ) -> dict[str, dict]:
        """
        Using LLM, ideintifies if the current conference part contains conference talk, 
        question section or both talk and question section. 
        Returns dictionary where:
        - conference_part_type (Literal["conference_talk", "questions_section", 
        "both_talk_and_questions_section"]) – type of this part of the conference;
        - start_time (List[float]) – time when the conference talk(s) or question section begin(s);
        - end_time (List[float]) – time when the conference talk(s) or question section end(s).
        """
        context = str(context)
        parser = PydanticOutputParser(pydantic_object=TypeIdentifier)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("user", self.user_prompt),
            ]
        ).partial(format_instructions=parser.get_format_instructions())
        chain = prompt | self.llm
        print("PROMPT:", prompt.invoke({"context": context}))
        response = chain.invoke({"context": context}).content
        return eval(response)
    
    def map_conference_talk_to_chunk(
            self,
            chunk_type_time : dict,
            chunk_with_corrected_diar : dict
    ) -> List[dict]:
        """
        Maps conference part and its boundaries to the chunk
        """
        talks_in_chunk = []
        for start_time, end_time in zip(chunk_type_time["startTime"], chunk_type_time["endTime"]):
            print(f"Start time of the conference part: {start_time}; end time of the conference part: {end_time}")
            current_talk = {
                "conference_part_type": chunk_type_time["conference_part_type"],
                "conference_part": []
                }
            previous_utterance = None
            previous_added = False
            for utterance in chunk_with_corrected_diar:
                if utterance["endTime"] <= start_time:
                    previous_utterance = utterance
                elif utterance["startTime"] <= end_time:
                    if previous_utterance and not previous_added:
                        current_talk["conference_part"].append(previous_utterance)
                        previous_added = True
                    current_talk["conference_part"].append(utterance)
            talks_in_chunk.append(current_talk)
        return talks_in_chunk
    
    def get_conference_talks(
            self,
            chunks_with_corrected_diar : List[dict]
    ) -> List[dict]:
        """
        Identifies type of the conference part 
        (either conference talk or questions section)
        and time boundaries of the different talks / questions inside it
        """
        conference_talks = []
        for chunk_with_corrected_diar in tqdm(chunks_with_corrected_diar,
                                              desc="Processing conference..."):
            chunk_type = self.identify_part_type(chunk_with_corrected_diar)
            current_chunk_talks = self.map_conference_talk_to_chunk(chunk_type, chunk_with_corrected_diar)
            conference_talks.extend(current_chunk_talks)
        return conference_talks