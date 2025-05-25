from longdocpipeline.pipeline.tasks.ner_unique import remove_duplicates


class Postprocessor:
    def __init__(
            self,
            task: str,
    ):
        self.task = task

    def __call__(self, text: str) -> str:
        if self.task == "ner_unique":
            text = remove_duplicates(text)
        return text
