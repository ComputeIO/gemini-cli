# External LLM Providers Configuration

The Gemini CLI now supports multiple LLM providers beyond Google's Gemini models. This guide explains how to configure and use external providers like Ollama, OpenAI, and other OpenAI-compatible APIs.

## Supported Providers

### 1. Ollama (Local)
Run open-source models locally using Ollama.

**Prerequisites:**
- Install [Ollama](https://ollama.ai) on your system
- Pull a model: `ollama pull deepseek-r1:latest`

**Configuration:**
```bash
# Optional: Set custom Ollama base URL (defaults to http://localhost:11434)
export OLLAMA_BASE_URL="http://localhost:11434"

# Optional: Set API key if your Ollama instance requires authentication
export OLLAMA_API_KEY="your-ollama-api-key"
```

**Usage:**
```bash
# Run Gemini CLI with Ollama
gemini --auth-type ollama --model deepseek-r1:latest
```

### 2. OpenAI
Use OpenAI's models like GPT-4, GPT-3.5-turbo, etc.

**Configuration:**
```bash
# Set your OpenAI API key
export OPENAI_API_KEY="sk-your-openai-api-key"

# Optional: Set custom base URL (defaults to https://api.openai.com/v1)
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

**Usage:**
```bash
# Run Gemini CLI with OpenAI
gemini --auth-type openai --model gpt-4
```

### 3. Custom OpenAI-Compatible APIs
Use any OpenAI-compatible API provider (e.g., Azure OpenAI, Anthropic via proxy, local APIs).

**Configuration:**
```bash
# Set your API key
export CUSTOM_LLM_API_KEY="your-api-key"

# Set the base URL of your API
export CUSTOM_LLM_BASE_URL="https://your-api-endpoint.com/v1"

# Optional: Set custom headers if needed
export CUSTOM_LLM_HEADERS='{"X-Custom-Header": "value"}'
```

**Usage:**
```bash
# Run Gemini CLI with custom provider
gemini --auth-type custom-openai-compatible --model your-model-name
```

## Authentication Methods

The CLI supports these authentication methods:

- `oauth-personal` - Login with Google (default)
- `gemini-api-key` - Use Gemini API key
- `vertex-ai` - Use Vertex AI
- `cloud-shell` - Use Cloud Shell credentials
- `ollama` - Use Ollama (local)
- `openai` - Use OpenAI API
- `custom-openai-compatible` - Use custom OpenAI-compatible API

## Environment Variables Reference

### Ollama
| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OLLAMA_BASE_URL` | No | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_API_KEY` | No | - | API key if authentication is required |

### OpenAI
| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | - | Your OpenAI API key |
| `OPENAI_BASE_URL` | No | `https://api.openai.com/v1` | OpenAI API base URL |

### Custom OpenAI-Compatible
| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `CUSTOM_LLM_API_KEY` | Yes | - | Your API key |
| `CUSTOM_LLM_BASE_URL` | Yes | - | API base URL |
| `CUSTOM_LLM_HEADERS` | No | - | JSON string of custom headers |

## Model Configuration

### Default Models
- **Ollama**: `deepseek-r1:latest`
- **OpenAI**: `gpt-4`
- **Custom**: Uses the model specified in CLI

### Setting Custom Models
You can specify models using the `--model` flag:

```bash
# For Ollama
gemini --auth-type ollama --model llama2:13b

# For OpenAI
gemini --auth-type openai --model gpt-3.5-turbo

# For custom providers
gemini --auth-type custom-openai-compatible --model claude-3-sonnet
```

## Examples

### Using DeepSeek-R1 with Ollama
```bash
# 1. Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 2. Pull DeepSeek-R1 model
ollama pull deepseek-r1:latest

# 3. Run Gemini CLI
export OLLAMA_BASE_URL="http://localhost:11434"
gemini --auth-type ollama --model deepseek-r1:latest
```

### Using OpenAI GPT-4
```bash
# 1. Set your OpenAI API key
export OPENAI_API_KEY="sk-your-api-key"

# 2. Run Gemini CLI
gemini --auth-type openai --model gpt-4
```

### Using Azure OpenAI
```bash
# 1. Set your Azure credentials
export CUSTOM_LLM_API_KEY="your-azure-api-key"
export CUSTOM_LLM_BASE_URL="https://your-resource.openai.azure.com/openai/deployments/your-deployment"
export CUSTOM_LLM_HEADERS='{"api-version": "2023-12-01-preview"}'

# 2. Run Gemini CLI
gemini --auth-type custom-openai-compatible --model gpt-4
```

## Troubleshooting

### Ollama Issues
1. **Connection refused**: Make sure Ollama is running (`ollama serve`)
2. **Model not found**: Pull the model first (`ollama pull model-name`)
3. **Permission denied**: Check if Ollama is accessible on the specified port

### OpenAI Issues
1. **Authentication failed**: Verify your API key is correct
2. **Rate limits**: Check your OpenAI usage and billing status
3. **Model access**: Ensure you have access to the requested model

### Custom API Issues
1. **Invalid base URL**: Ensure the URL is correct and accessible
2. **Authentication failed**: Check API key and any required headers
3. **Model compatibility**: Verify the API follows OpenAI's specification

## Feature Compatibility

| Feature | Gemini | Ollama | OpenAI | Custom |
|---------|--------|--------|--------|--------|
| Text Generation | ✅ | ✅ | ✅ | ✅ |
| Streaming | ✅ | ✅ | ✅ | ✅ |
| Token Counting | ✅ | ~* | ✅ | ~* |
| Embeddings | ✅ | ~* | ✅ | ~* |
| Function Calling | ✅ | ⚠️** | ✅ | ⚠️** |

*Approximate token counting used for providers without dedicated endpoints
**Function calling depends on the specific model and API support

## Migration from Gemini

When switching from Gemini to another provider, note:

1. **Model names**: Different providers use different model naming conventions
2. **Rate limits**: Each provider has different rate limiting policies
3. **Cost**: Pricing varies significantly between providers
4. **Features**: Not all providers support all Gemini features (e.g., function calling)
5. **Response format**: The CLI handles format conversion automatically
