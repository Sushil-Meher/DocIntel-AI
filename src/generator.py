from transformers import pipeline


generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    device = 0
)

def generate_answer(prompt: str) -> str:

    result = generator(
        prompt,
        max_new_tokens=100
    )

    return result[0]["generated_text"]




