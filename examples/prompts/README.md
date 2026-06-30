# Example System Prompts

Plain-text system prompts you can load with `--prompt-file`:

```bash
./rkllm_server model.rkllm --prompt-file examples/prompts/customer_support.txt
```

The file content is wrapped automatically with `<|im_start|>system\n...\n<|im_end|>\n`
at startup — write plain text, no chat-template markers needed.

## Files

| File | Use case | Output schema |
|---|---|---|
| `customer_support.txt` | Classify support tickets + extract entities | `{intent, entities}` |
| `e_commerce_order.txt` | Parse natural-language order requests | `{intent, items[]}` |

## Compatibility with `/process`

`/process` validates only that the output is a JSON object with a string `intent`
field. Both example prompts above satisfy that and benefit from adaptive
recovery — just swap them in with `--prompt-file`.

If your output schema doesn't fit `{intent: "..."}`, use `/generate` instead —
it returns raw output with no validation:

```bash
curl -X POST http://localhost:18080/generate -H "Content-Type: application/json" -d '{
  "system_prompt": "...inline prompt...",
  "user_text": "..."
}'
```

For stricter schema validation (e.g. enum-bound intent, required nested fields),
fork `src/main.cpp` and adapt `is_valid_intent_response()` / `process_with_recovery()`.
A configurable JSON-Schema validator is on the roadmap.

## Writing a good prompt for small NPU models

Tested with Qwen2.5-1.5B w8a8 on RK3588. What works:

- **Short, declarative rules** — bullet points beat prose.
- **3–5 few-shot examples** covering the edge cases you care about.
- **Explicit "respond with JSON in one line"** — small models love to add prose otherwise.
- **Negative examples** when a class is consistently misclassified (e.g. "do NOT
  multiply numbers without units").
- **Defaults stated upfront** so the model fills missing fields predictably.

What doesn't work:

- Long system prompts (>1500 tokens) burn prefill latency badly.
- "Be helpful and concise" type meta-instructions — wasted tokens on 1.5B models.
- Asking for prose + JSON together — the model picks one and ignores the other.
