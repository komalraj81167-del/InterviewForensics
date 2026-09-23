from backend.app.transcription import (
    transcribe_audio_with_word_timestamps
)
from backend.app.interview_segmentation import (
    detect_questions,
    detect_answer_boundaries,
    build_interview_turns,
    validate_questions
)

file_path = "backend/uploads/mock_interview_01.mp3"


print("\n1. Getting word-level timestamps...")

word_segments = transcribe_audio_with_word_timestamps(
    file_path
)

print("Total words:", len(word_segments))


print("\n2. Detecting interview questions...")

question_result = detect_questions(
    word_segments
)
print("\n3. Refining answer boundaries...")

question_result = detect_answer_boundaries(
    word_segments,
    question_result
)

print("\nRAW QUESTION DETECTION")
print("======================")

print(question_result)


print("\n4. Validating questions...")

validation = validate_questions(
    question_result,
    len(word_segments)
)


print("\nQUESTION VALIDATION")
print("===================")

print("Valid:", validation["valid"])


if validation["errors"]:

    print("\nErrors:")

    for error in validation["errors"]:
        print("-", error)

else:

    print("No question segmentation errors found.")


print("\n5. Building question-answer pairs...")

turns = build_interview_turns(
    word_segments,
    question_result
)
print("\nDETECTED QUESTION COUNT")
print("=======================")

print(
    "Total questions:",
    len(turns)
)

print("\nINTERVIEW SEGMENTATION")
print("======================")

for i, turn in enumerate(
    turns,
    start=1
):

    print(f"\nQUESTION {i}")
    print("----------")

    print(turn["question"])

    print(
        f"\nQuestion time: "
        f"{turn['question_start']:.2f}s - "
        f"{turn['question_end']:.2f}s"
    )


    print("\nANSWER")
    print("------")

    print(turn["answer"])


    print(
        f"\nAnswer time: "
        f"{turn['answer_start']:.2f}s - "
        f"{turn['answer_end']:.2f}s"
    )