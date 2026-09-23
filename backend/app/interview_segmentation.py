from dotenv import load_dotenv
from google import genai
from google.genai import types

import os
import json
import time


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not set in .env")


client = genai.Client(api_key=api_key)


def detect_questions(word_segments):
    """
    Detect only the actual interviewer questions.

    Gemini identifies the word IDs belonging to each question.
    Python will construct the answers later.
    """

    numbered_words = []

    for i, word in enumerate(word_segments):

        numbered_words.append({
            "id": i,
            "start": word["start"],
            "end": word["end"],
            "word": word["word"]
        })


    words_text = json.dumps(
        numbered_words,
        indent=2
    )


    prompt = f"""
You are an expert interview transcript analyzer.

This is a simulated technical interview.

IMPORTANT:
- There may be only ONE physical speaker.
- The same person may speak both interviewer questions
  and candidate answers.
- Do NOT use speaker identity.
- Identify questions based on meaning, grammar,
  conversational context, and pauses.

Your ONLY task is to identify the actual interviewer
questions in the transcript.

Do NOT identify answers.

For every interviewer question:

1. Identify the exact word ID where the question starts.
2. Identify the exact word ID where the question ends.
3. Preserve the original transcript wording.
4. Do NOT rewrite or correct words.
5. A question may contain multiple sentences.
6. Ignore greetings and conversational feedback such as:
   "okay", "that's great", "sure", "alright", etc.
7. Do not treat a candidate's statement as a question.
8. Do not split one question into multiple questions.
9. Do not create questions from fragments of candidate answers.
10. Only include genuine interview questions.

Examples of conversational feedback that should NOT
be considered questions:

"okay"
"that's great"
"sure"
"alright"
"good"
"okay that's great"

Examples of actual interview questions:

"tell me about yourself"
"explain one project you have worked on"
"what was your contribution to the project"
"explain the difference between a process and a thread"
"tell me about a difficult problem you faced and how you solved it"
"why should we hire you"

IMPORTANT:
Whisper transcription may contain recognition errors.

DO NOT correct the transcript.

Return ONLY valid JSON.

Required format:

{{
    "questions": [
        {{
            "question_start_id": 0,
            "question_end_id": 5
        }}
    ]
}}

Return ONLY word IDs.

Here is the word-level transcript:

{words_text}
"""


    max_retries = 3


    for attempt in range(max_retries):

        try:

            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json"
                )
            )

            break


        except Exception:

            if attempt == max_retries - 1:
                raise

            wait_time = 2 ** attempt

            print(
                f"Gemini request failed. "
                f"Retrying in {wait_time} seconds..."
            )

            time.sleep(wait_time)


    result = response.text.strip()

    return json.loads(result)

def detect_answer_boundaries(word_segments, question_result):
    """
    Determine where each candidate answer ends.

    Question boundaries are already detected by detect_questions().
    Gemini now examines the text between questions and identifies
    which words belong to the candidate's answer.

    The goal is to remove interviewer-style transition/feedback
    that occurs before the next question.
    """

    questions = question_result.get(
        "questions",
        []
    )

    refined_questions = []

    for i, question in enumerate(questions):

        question_start_id = question[
            "question_start_id"
        ]

        question_end_id = question[
            "question_end_id"
        ]

        # --------------------------------------------------
        # Determine the region after this question
        # and before the next question.
        # --------------------------------------------------

        answer_start_id = question_end_id + 1

        if i + 1 < len(questions):

            next_question_start_id = questions[
                i + 1
            ]["question_start_id"]

            region_end_id = next_question_start_id - 1

        else:

            region_end_id = len(word_segments) - 1

        if answer_start_id > region_end_id:

            refined_questions.append({
                "question_start_id": question_start_id,
                "question_end_id": question_end_id,
                "answer_start_id": answer_start_id,
                "answer_end_id": answer_start_id - 1
            })

            continue

        # --------------------------------------------------
        # Build numbered words for this region.
        # --------------------------------------------------

        region_words = []

        for word_id in range(
            answer_start_id,
            region_end_id + 1
        ):

            word = word_segments[word_id]

            region_words.append({
                "id": word_id,
                "start": word["start"],
                "end": word["end"],
                "word": word["word"]
            })

        region_text = json.dumps(
            region_words,
            indent=2
        )

        # --------------------------------------------------
        # Ask Gemini to find the candidate answer end.
        # --------------------------------------------------

        prompt = f"""
You are analyzing one section of a simulated interview.

There is only one physical speaker because the candidate
is also recording the interviewer questions.

The interview question has already been identified.

Your task is to identify the EXACT word ID where the
candidate's answer ends.

After the candidate finishes answering, there may be
interviewer-style transition or feedback such as:

- okay
- okay that's great
- that's great
- sure
- alright
- then
- thank you
- short conversational remarks

These should NOT be included in the candidate answer.

IMPORTANT:

1. Preserve the original transcript.
2. Do NOT correct Whisper transcription errors.
3. Do NOT rewrite words.
4. Do NOT summarize the answer.
5. The answer must begin at the provided answer_start_id.
6. The answer must end before obvious interviewer
   transition/feedback when such transition exists.
7. If there is no obvious interviewer transition,
   the answer can end at the final word in the region.
8. Return ONLY the answer_end_id.
9. The returned ID must exist in the supplied word list.

Interview question:

{json.dumps(
    " ".join(
        word_segments[j]["word"]
        for j in range(
            question_start_id,
            question_end_id + 1
        )
    )
)}

Answer candidate region:

{region_text}

Return exactly this JSON format:

{{
    "answer_end_id": 123
}}
"""

        max_retries = 3

        for attempt in range(max_retries):

            try:

                response = client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        response_mime_type="application/json"
                    )
                )

                break

            except Exception:

                if attempt == max_retries - 1:
                    raise

                wait_time = 2 ** attempt

                print(
                    f"Gemini answer-boundary request failed. "
                    f"Retrying in {wait_time} seconds..."
                )

                time.sleep(wait_time)

        result = response.text.strip()

        boundary = json.loads(result)

        answer_end_id = boundary.get(
            "answer_end_id"
        )

        # --------------------------------------------------
        # Safety validation
        # --------------------------------------------------

        if answer_end_id is None:

            answer_end_id = region_end_id

        if (
            answer_end_id < answer_start_id
            or answer_end_id > region_end_id
        ):

            answer_end_id = region_end_id

        refined_questions.append({
            "question_start_id": question_start_id,
            "question_end_id": question_end_id,
            "answer_start_id": answer_start_id,
            "answer_end_id": answer_end_id
        })

    return {
        "questions": refined_questions
    }
def build_interview_turns(word_segments, question_result):
    """
    Build final question-answer pairs using the refined
    question and answer boundaries.
    """

    questions = question_result.get(
        "questions",
        []
    )

    turns = []

    for question in questions:

        question_start_id = question[
            "question_start_id"
        ]

        question_end_id = question[
            "question_end_id"
        ]

        answer_start_id = question.get(
            "answer_start_id"
        )

        answer_end_id = question.get(
            "answer_end_id"
        )

        question_words = word_segments[
            question_start_id:question_end_id + 1
        ]

        if (
            answer_start_id is not None
            and answer_end_id is not None
            and answer_start_id <= answer_end_id
        ):

            answer_words = word_segments[
                answer_start_id:answer_end_id + 1
            ]

        else:

            answer_words = []

        if not question_words:
            continue

        question_text = " ".join(
            word["word"]
            for word in question_words
        ).strip()

        answer_text = " ".join(
            word["word"]
            for word in answer_words
        ).strip()

        turns.append({
            "question": question_text,
            "answer": answer_text,

            "question_start": question_words[0]["start"],
            "question_end": question_words[-1]["end"],

            "answer_start": (
                answer_words[0]["start"]
                if answer_words
                else None
            ),

            "answer_end": (
                answer_words[-1]["end"]
                if answer_words
                else None
            )
        })

    return turns  
def validate_questions(question_result, total_words):
    """
    Validate detected question ranges.
    """

    errors = []

    questions = question_result.get(
        "questions",
        []
    )


    previous_end_id = -1


    for i, question in enumerate(
        questions,
        start=1
    ):

        start_id = question.get(
            "question_start_id"
        )

        end_id = question.get(
            "question_end_id"
        )


        if start_id is None or end_id is None:

            errors.append(
                f"Question {i}: Missing word ID."
            )

            continue


        if start_id < 0 or start_id >= total_words:

            errors.append(
                f"Question {i}: Start ID "
                f"{start_id} is outside valid range."
            )


        if end_id < 0 or end_id >= total_words:

            errors.append(
                f"Question {i}: End ID "
                f"{end_id} is outside valid range."
            )


        if start_id > end_id:

            errors.append(
                f"Question {i}: Start ID is "
                f"after end ID."
            )


        if start_id <= previous_end_id:

            errors.append(
                f"Question {i}: Overlaps with "
                f"previous question."
            )


        previous_end_id = end_id


    return {
        "valid": len(errors) == 0,
        "errors": errors
    }