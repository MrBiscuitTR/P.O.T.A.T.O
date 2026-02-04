# Model Tag Detection System

Auto-detects thinking tags and output formats for different LLM models.

## Usage

### Detect tags for a specific model:
```bash
cd POTATO
python -m components.utilities.initial_model_inference_setup --model "glm-4.7-flash:Q4_K_M"
```

### Detect tags for all models:
```bash
python -m components.utilities.initial_model_inference_setup --all
```

## How it works

1. **Test Inference**: Runs a reasoning test on the target model
2. **Analysis**: Uses qwen2.5-coder:7b to analyze the output structure
3. **Tag Detection**: Identifies thinking tags (e.g., `<think>`, `<reasoning>`, or custom tags)
4. **Storage**: Saves detected tags to `.data/.modelinfos.json`

## Detected Information

For each model, the script detects:
- `thinking_tags`: Opening and closing tags for reasoning
- `has_inline_thinking`: Whether thinking appears inline or separate
- `supports_streaming_thinking`: Whether the model can stream thoughts
- `detection_notes`: Any special observations

## Model Info Format

```json
{
  "model-name": {
    "can_think": true,
    "tool_calls": true,
    "vision": false,
    "thinking_tags": {
      "open": "<think>",
      "close": "</think>"
    },
    "has_inline_thinking": false,
    "supports_streaming_thinking": true,
    "detection_notes": "Uses standard think tags"
  }
}
```

## Default Behavior

- Default tags: `<think>...</think>`
- Models without detected tags use defaults
- First-time usage triggers detection automatically (future feature)

## Re-detection

To re-detect tags for a model (e.g., after model update):
```bash
python -m components.utilities.initial_model_inference_setup --model "model-name"
```

## Dependencies

- `ollama` Python library
- `qwen2.5-coder:7b` model (for analysis)
- Target model to be tested
