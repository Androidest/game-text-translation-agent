from typing_extensions import TypedDict, List

class TermExtractionState(TypedDict):
    input_list: List[str]
    messages: List
    last_response: List[List[str]]
    error: str
    retry_count: int
    output_list: List[List[str]]