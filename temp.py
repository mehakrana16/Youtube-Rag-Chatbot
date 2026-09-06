from src.llm import answer_question

fake_chunks = [
    {"text": "The video discusses how Alia Bhatt manages her ADHD diagnosis and anxiety.", "start_time": 120.0},
    {"text": "She talks about finding peace through meditation and journaling as a new mother.", "start_time": 340.5},
]

answer = answer_question("What is the capital of Japan?", fake_chunks)
print(answer)
