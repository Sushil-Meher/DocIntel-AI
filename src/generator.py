from transformers import pipeline


generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    device=0
)


def generate_answer(prompt: str) -> str:

    result = generator(
        prompt,
        max_new_tokens=120,
        do_sample=False,
        return_full_text=False,
        clean_up_tokenization_spaces=False
    )

    return result[0]["generated_text"].strip()