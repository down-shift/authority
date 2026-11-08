# Lesson: gpt-oss puts a content type in the final header when tools are listed

**Type:** lesson. **Audience:** every agent parsing gpt-oss (harmony) output.

On 2026-10-16 the T3 sweep (P2.8) started under `agent-v2`. 38% of the first
4,096 gpt-oss-120b episodes came back as empty first turns, which the harness
scored as protocol failures. The model had answered correctly. When the prompt
lists tools, gpt-oss often writes the final channel as
`<|channel|>final <|constrain|>commentary<|message|>{...}` (also `json` or
`JSON`), and `split_harmony` matched only `final<|message|>`. T2 prompts list
no tools, and the T2 raw outputs contain no `<|constrain|>` token.

The episode record kept only the extracted text, so finding the cause took an
extra GPU diagnostic to regenerate raw outputs.

**How to apply:**
- For harmony models, read raw outputs (with special tokens) from the first
  batch of any new prompt format, not just the outcome counts. An empty answer
  with `finish_reason: stop` and a short output means extraction, not the model.
- Header grammar: `<|channel|>final` may be followed by
  `<|constrain|><type>` before `<|message|>` (fixed in `agent-v3`).
- A fix after seeing data is a version bump, a dated prereg amendment, and a
  rerun of every affected run (`docs/prereg/phase2-t3.md` § Amendments).

See [[lesson-qwen38-serving-traps]] (the same failure shape for think blocks).
