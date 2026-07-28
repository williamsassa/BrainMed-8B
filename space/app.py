"""Gradio demo for BrainMed-8B.

The model is trained to reason inside <think>...</think> and answer inside <answer>...</answer>
under a fixed system prompt. The UI keeps that contract: the same system prompt is sent at
inference (dropping it is a train/serve mismatch that measurably costs accuracy), and the two
blocks are shown separately - the reasoning is visible but clearly marked as the model's
working, not as an answer.
"""
import os
import re
import threading

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MODEL_ID = os.environ.get("MODEL_ID", "BrainHealthAI/BrainMed-8B")
TOKEN = os.environ.get("HF_TOKEN")          # required while the repo is private

SYSTEM = ("You are a medical reasoning assistant. Work through the clinical problem step by step "
          "inside <think>...</think>, grounding every step in established medical knowledge, then "
          "give the final, complete answer inside <answer>...</answer>.")

DISCLAIMER = """
> **Research demo — not medical advice.** This model is a fine-tuned language model evaluated
> on multiple-choice benchmarks. It is not a medical device, has not been clinically validated,
> and must not be used to diagnose, treat, or make decisions about any real person.
> **In an emergency, call your local emergency number.**
"""

EXAMPLES = [
    "A 58-year-old man with type 2 diabetes presents with sudden crushing chest pain radiating to "
    "the left arm, diaphoresis and nausea for 40 minutes. What is your immediate management?",
    "A 7-year-old has had fever for 6 days, bilateral non-purulent conjunctivitis, strawberry "
    "tongue, cracked lips, a polymorphous rash and unilateral cervical lymphadenopathy. Most "
    "likely diagnosis, and what must be ruled out urgently?",
    "Please answer the following multiple-choice questions, ensuring your response concludes with "
    "the correct option in the format: 'The answer is A.'.\n"
    "A 25-year-old primigravida at 36 weeks presents with severe headache, hypertension, oedema "
    "and proteinuria, then has a generalized tonic-clonic seizure. Which agent controls the "
    "seizures?\nA. Phenytoin\nB. Magnesium sulfate\nC. Diazepam\nD. Levetiracetam",
    "Explain the mechanism by which ACE inhibitors cause a dry cough, and what to switch to.",
]

# ZeroGPU (HF PRO / Team) hands a GPU slice to decorated functions only; elsewhere the
# decorator is a no-op and the model simply sits on whatever device is available.
try:
    import spaces
    ZEROGPU = True
except ImportError:                                        # noqa: BLE001
    ZEROGPU = False

    class _NoSpaces:
        @staticmethod
        def GPU(*a, **k):
            def wrap(fn):
                return fn
            return wrap
    spaces = _NoSpaces()

print(f"loading {MODEL_ID} ... (zerogpu={ZEROGPU})", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=TOKEN, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, token=TOKEN, trust_remote_code=True, torch_dtype=torch.bfloat16,
    device_map=None if ZEROGPU else ("auto" if torch.cuda.is_available() else None),
)
model.eval()
if not ZEROGPU and not torch.cuda.is_available():
    print("WARNING: no GPU. An 8B model needs ~16 GB of VRAM; on CPU this will be unusable "
          "and may run out of memory. Assign GPU hardware in the Space settings.", flush=True)
print(f"loaded (cuda available: {torch.cuda.is_available()})", flush=True)

THINK = re.compile(r"<think>(.*?)(?:</think>|$)", re.S)
ANSWER = re.compile(r"<answer>(.*?)(?:</answer>|$)", re.S)


def split(text):
    t, a = THINK.search(text), ANSWER.search(text)
    if not t and not a:
        return "", text          # model answered without the tags
    return (t.group(1).strip() if t else ""), (a.group(1).strip() if a else "")


@spaces.GPU(duration=120)
def respond(message, history, system_prompt, max_new_tokens, temperature):
    if not message or not message.strip():
        yield "", ""
        return
    if ZEROGPU and next(model.parameters()).device.type != "cuda":
        model.to("cuda")          # ZeroGPU only exposes the device inside this call
    msgs = [{"role": "system", "content": system_prompt}] if system_prompt.strip() else []
    for turn in history or []:
        # gradio 5 passes history as a list of {role, content} dicts
        if isinstance(turn, dict):
            msgs.append(turn)
    msgs.append({"role": "user", "content": message})

    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    kwargs = dict(**inputs, streamer=streamer, max_new_tokens=int(max_new_tokens),
                  pad_token_id=tokenizer.eos_token_id)
    if temperature and temperature > 0:
        kwargs.update(do_sample=True, temperature=float(temperature), top_p=0.9)
    else:
        kwargs.update(do_sample=False)

    threading.Thread(target=model.generate, kwargs=kwargs, daemon=True).start()
    acc = ""
    for chunk in streamer:
        acc += chunk
        reasoning, answer = split(acc)
        yield answer, reasoning


with gr.Blocks(title="BrainMed-8B", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# BrainMed-8B\nMedical reasoning model — full fine-tune of HuatuoGPT-o1-8B.")
    gr.Markdown(DISCLAIMER)

    with gr.Row():
        with gr.Column(scale=3):
            question = gr.Textbox(label="Clinical question", lines=5,
                                  placeholder="Describe the case, or paste a multiple-choice question…")
            with gr.Row():
                send = gr.Button("Ask", variant="primary")
                clear = gr.Button("Clear")
            answer = gr.Textbox(label="Answer", lines=10, show_copy_button=True)
            with gr.Accordion("Model's reasoning (its working, not a conclusion)", open=False):
                reasoning = gr.Textbox(label="", lines=16, show_copy_button=True)
        with gr.Column(scale=1):
            system_prompt = gr.Textbox(label="System prompt", value=SYSTEM, lines=6,
                                       info="Matches training. Changing it degrades accuracy.")
            max_new_tokens = gr.Slider(128, 2048, value=900, step=64, label="Max new tokens")
            temperature = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Temperature",
                                    info="0 = greedy, as used for the published scores")

    gr.Examples(examples=[[e] for e in EXAMPLES], inputs=[question], label="Examples")

    send.click(respond, [question, gr.State([]), system_prompt, max_new_tokens, temperature],
               [answer, reasoning])
    question.submit(respond, [question, gr.State([]), system_prompt, max_new_tokens, temperature],
                    [answer, reasoning])
    clear.click(lambda: ("", "", ""), None, [question, answer, reasoning])

if __name__ == "__main__":
    # GRADIO_SHARE=1 opens a public tunnel - the way to demo this from a GPU box you already
    # rent (a training pod), without paying for Space hardware. The link expires after 72h.
    demo.queue(max_size=16).launch(
        share=os.environ.get("GRADIO_SHARE", "0") == "1",
        server_name=os.environ.get("GRADIO_HOST", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_PORT", "7860")),
    )
