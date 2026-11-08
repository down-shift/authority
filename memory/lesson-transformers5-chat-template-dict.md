# Lesson: in transformers 5, `apply_chat_template(tokenize=True)` returns a dict

**Type:** lesson. **Audience:** every agent counting or building tokens with a chat template.

In P4.1 (2026-11-07) the SFT builder counted tokens with
`len(tok.apply_chat_template(messages, tokenize=True))`. Under transformers 5.17
that call returns an encoding (`input_ids`, `attention_mask`), so `len()` was 2
for every example. The token-matched control set was matched on example count, not
tokens, and nothing crashed. The first full build was discarded before anything used it.

**How to apply:**
- Read the ids explicitly: `apply_chat_template(..., tokenize=True, return_dict=True)["input_ids"]`.
  Unwrap a batch dimension if there is one.
- Guard any count you depend on with a plausibility check. The builder refuses a build
  averaging under 50 tokens per example; a prompt-v1 query is hundreds of tokens.
- Before trusting a token count, check one example by hand against its character
  length (≈3–4 characters per token for these prompts).
