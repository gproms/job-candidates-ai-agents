from typing import TypedDict, Dict, List

class AgentState(TypedDict):
    cv_data: Dict[str, Dict[str, str]]
    linkedin_data: Dict[str, Dict[str, List[str]]]
    interview_data: Dict[str, Dict[str, Dict[str, str]]]
    profiles: Dict[str, Dict[str, Dict[str, str]]]
