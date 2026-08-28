# OpenAI-compatible model: native list tool return content

Subclassed `OpenAIChatModel` as `OpenAICompatibleModel` to send list-type
`ToolReturnPart` content as native `list[ChatCompletionContentPartTextParam]`
instead of a JSON-serialized string. This fixes compatibility with
OpenAI-compatible models (e.g. GLM-5) whose chat templates branch on `tool`
role `content` being a string vs list.

All OpenAI-compatible providers (`deepseek:`, `grok:`, `openrouter:`,
`perplexity:`, `lm-studio:`, `zen:`, `copilot:`) now use
`OpenAICompatibleModel` via `_get_openai_based_model()`. Non-list content,
multimodal file content, and failed tool returns fall back to the parent's
string serialization.

Closes #112.
