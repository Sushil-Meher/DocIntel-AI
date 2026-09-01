from transformers import pipeline


generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    device=0
)

# The pipeline builds its own merged generation_config with a leftover
# max_length=20 default, even though the model's own config has no
# max_length set. That stale value fights with max_new_tokens below and
# triggers "Both max_new_tokens and max_length seem to have been set."
generator.generation_config.max_length = None


def generate_answer(prompt: str) -> str:

    result = generator(
        prompt,
        max_new_tokens=120,
        do_sample=False,
        return_full_text=False,
        clean_up_tokenization_spaces=False
    )

    return result[0]["generated_text"].strip()