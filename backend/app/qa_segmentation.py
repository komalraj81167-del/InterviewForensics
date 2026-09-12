import re


def split_into_sentences(text):
    """Split transcript into sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    return [sentence.strip() for sentence in sentences if sentence.strip()]


def is_question(sentence):
    """Check whether a sentence looks like a question."""

    question_patterns = [
        r'\?$',
        r'^(what|why|how|when|where|who|which|can|could|would|will|do|did|have|has|tell|explain|describe|are|is)\b'
    ]

    sentence_lower = sentence.lower().strip()

    for pattern in question_patterns:
        if re.search(pattern, sentence_lower):
            return True

    return False


def segment_interview(transcript):
    """Split transcript into questions and answers."""

    sentences = split_into_sentences(transcript)

    segments = []
    current_question = None
    current_answer = []

    for sentence in sentences:

        if is_question(sentence):

            # Save previous question-answer pair
            if current_question:
                segments.append({
                    "question": current_question,
                    "answer": " ".join(current_answer).strip()
                })

            current_question = sentence
            current_answer = []

        else:
            if current_question:
                current_answer.append(sentence)

    # Save final question-answer pair
    if current_question:
        segments.append({
            "question": current_question,
            "answer": " ".join(current_answer).strip()
        })

    return segments