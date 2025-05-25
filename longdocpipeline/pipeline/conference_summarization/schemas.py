from pydantic import BaseModel, Field
from typing import Literal, List


class TypeIdentifier(BaseModel):
    """
    Ideintifies if the current conference part contains conference talk, 
    question section or both talk and question section.
    There can be more than one conference talk in the conference part.
    Also returns the time when the conference talk(s) or the questions section(s) begin(s),
    and the time when the conference talk(s) of the questions section(s) end(s).
    Length of the list startTime MUST BE EQUAL to length of the list endTime!
    startTime MUST BE EQUAL to STARTTIME OF THE UTTERANCE in the current conference part!
    endTime MUST BE EQUAL to ENDTIME OF THE UTTERANCE in the current conference part!
    """
    conference_part_type : Literal["conference_talk", "questions_section", "both_talk_and_questions_section"] = Field(
        description="Type of this part of the conferece: either conference_talk, questions_section or (RARELY) both conference talk and question section"
        )
    startTime : List[float] = Field(
        description="Time when the conference talk(s) or question section begin(s)"
    )
    endTime : List[float] = Field(
        description="Time when the conference talk(s) or question section end(s)"
    )


class ConferenceTalkStructure(BaseModel):
    """
    Returns the number of the main speaker in this part of the conference,
    as well as their name, the topics of the conference speech and
    the main ideas and conclusions of the speech.
    """
    speaker_number: int = Field(description="Number of the main speaker")
    speaker_name: str = Field(description="Name of the main speaker")
    topic: str = Field(description="Topic of this conference speech")
    main_conclusions: str = Field(description="Main ideas and conclusions of this conference speech")


class FinalSummaryStructure(BaseModel):
    """
    Returns final summary in a form of the list of the summaries.
    """
    final_summary: List = Field(description="List of summaries")