# prompts.py

CV_PROMPT = """
You're excellent at extracting information, while being very honest, you never make up information when tasked on extracting information. You are now tasked with extracting the following details from the CV text, but do NOT make up any information! If you don't have any information, leave the field empty, or state unknown, but never ever make up information you didn't read on the text:
1. Candidate's name.
2. List of work experiences (title, company, duration_years (in int or float rather than str), description).
3. Academic qualifications (degree, institution, year).
4. List of skills.
Be careful not to confuse company names (work experience) with university name (education). Respond in JSON format with fields: "name", "Experience", "Education", "Skills". If you don't have any information, leave the field empty.
"""

LINKEDIN_PROMPT = """
You're excellent at extracting information, while being very honest, you never make up information when tasked on extracting information. You are now tasked with extracting the following details from the linkedin profile, but do NOT make up any information! If you don't have any information, leave the field empty, or state unknown, but never ever make up information you didn't read on the linkedin json context:
1. Candidate's name.
2. List of work experiences (title, company, duration_years (in int or float rather than str), description).
3. Academic qualifications (degree, institution, year).
4. List of skills.
Respond in JSON format with fields: "name", "Experience", "Education", "Skills". If you don't have any information, leave the field empty.
"""

#"""Please summarize the interview in the following JSON format:
INTERVIEW_PROMPT = """
You're excellent at extracting information, while being very honest, you never make up information when tasked on extracting information. You are now tasked with extracting the following details from the interview transcript, but do NOT make up any information! If you don't have any information, leave the field empty, or state unknown, but never ever make up information you didn't read on the text:
1. Candidate's name.
2. List of work experiences (title, company, duration_years (in int or float rather than str), description).
3. Academic qualifications (degree, institution, year).
4. List of skills.
Be careful not to confuse work experience with education. Respond in JSON format with fields: "name", "Experience", "Education", "Skills". If you don't have any information, leave the field empty or use unknown as value, remember that.
}"""

SYNTHESIS_PROMPT = """
You're excellent at combining information from multiple sources all coming in JSON format, while also being very honest, you never make up information when tasked on combining information from different JSON sources. You are now tasked with combining 3 JSON sources coming from the same job candidate applicant (CV, LinkedIn, Interview). When you find redundant information for a given field, extract the information from all the sources and merge them on a single one, ignoring the unknown values and focusing on known ones. Do NOT make up any information, leave the field empty if you don't have any information. Remember to deduplicate the contents on each field and merge redundant entries. The final output should have the following fields:
1. Merge work experiences. If there are redundant entries for the same job, combine them into a single summary.
2. Merge academic qualifications. If there are redundant entries for the same qualification, combine them in order to have one single entry per studies (where each entry can have its own degree, institution and year, deuplicating when necessary).
3. Merge skills (combine and deduplicate when necessary).
4. Summary: Generate a brief summary for this candidate, describing in one single sentence its work experiences, academic qualifications and top 3-5 most relevant skills.
Be careful not to confuse work experience with education (you can rely on the linkedin JSON to clarify what what is work experience or education, like company vs university names and other).
If you're adding multiple entries for work experience or education, verify that they do belong to different positions or studies, and not just repeated entries.
If repeated entries, merge them into a single one. You can also cross verify with the number of entries that appear on the CV or linkedin JSONs, to make sure you're not adding repeated entries.
This is particularly true if the number of work experience entries, or education entries, on the JSONs from CV or linkedin, differ with the number of entries on the final output, you can cross check to try to make them match in number of entries.
Make sure the duration_years is not in str format but rather in int or float.
Deduplicate any duplications you may find on any field, such as the lists of skills or anything else that may be repeated. This deduplication is case insensitive, so "Python" and "python" are considered the same skill and you can keep just one of them.
Respond in JSON format with fields: "name", "Summary", "Experience", "Education", "Skills". If you don't have any information, leave the field empty, and you can use unknown value only if the value was unknown on all of the sources.
"""

