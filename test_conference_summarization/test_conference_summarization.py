import sys
sys.path.append("/home/karysheva@ad.speechpro.com/Документы/diploma/structured_summarization")
#sys.path.append("/home/karysheva@ad.speechpro.com/Документы/LongDocs/gigachain_stc_structured_output/src")

import json

from longdocpipeline.pipeline.conference_summarization.conference_summarizer import ConferenceSummarizer
from constants import SYSTEM_PROMPT_PATH, USER_PROMPT_PATH, INPUT_PATH, PATH_TO_RESULTS


def test_conference_summarization():
    with open(SYSTEM_PROMPT_PATH, "r") as file:
        system_prompt_sum = file.read()
    with open(USER_PROMPT_PATH, "r") as file:
        user_prompt_sum = file.read()

    conference_summarizer = ConferenceSummarizer(system_prompt_sum, user_prompt_sum)
    summarized_conference = conference_summarizer.summarize_conference(INPUT_PATH)

    with open(PATH_TO_RESULTS, 'w', encoding='utf-8') as file:
        json.dump(summarized_conference.final_summary, file, ensure_ascii=False, indent=4)
    
    json_symbols = ["{", "}", "speaker_number", "speaker_name", "topic", "main_conclusions"]
    assert all(symbol in str(summarized_conference) for symbol in json_symbols) 


if __name__=="__main__":
    test_conference_summarization()