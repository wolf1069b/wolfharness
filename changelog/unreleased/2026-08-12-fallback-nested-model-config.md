# Parse nested fallback model configurations

`FallbackModelConfig` now accepts discriminated model configuration objects from YAML, including
custom OpenAI-compatible endpoints. This makes configured fallback chains behave the same whether
their sub-models are supplied as Python objects or manifest dictionaries.

String model configurations can also declare `api_key_env`, allowing each custom endpoint to
resolve its own runtime credential without placing secret values in a manifest.
