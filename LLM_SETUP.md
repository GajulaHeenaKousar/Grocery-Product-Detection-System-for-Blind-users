# LLM Setup Guide

## Overview

The system uses LLM (Large Language Model) to get product information when:
1. Barcode is detected but not found in Open Food Facts database
2. Object detection confidence is below 60%
3. Detected product is not in the training dataset

## Supported LLM Providers

### 1. OpenAI (Recommended)
- **Model**: GPT-4o-mini (falls back to GPT-3.5-turbo if not available)
- **Cost**: Low cost, good accuracy
- **Setup**: 
  ```bash
  set OPENAI_API_KEY=your_openai_api_key_here
  ```
- **Update config.py**: 
  ```python
  LLM_PROVIDER = "openai"
  ```

### 2. Anthropic Claude
- **Model**: Claude 3.5 Sonnet
- **Cost**: Slightly more expensive, very accurate
- **Setup**:
  ```bash
  set ANTHROPIC_API_KEY=your_anthropic_api_key_here
  ```
- **Update config.py**:
  ```python
  LLM_PROVIDER = "anthropic"
  ```

## Getting API Keys

### OpenAI API Key
1. Go to https://platform.openai.com/api-keys
2. Sign up or log in
3. Create a new API key
4. Copy the key and set it as environment variable

### Anthropic API Key
1. Go to https://console.anthropic.com/
2. Sign up or log in
3. Go to API Keys section
4. Create a new API key
5. Copy the key and set it as environment variable

## Configuration

### Step 1: Set API Key (Windows)
```bash
set OPENAI_API_KEY=sk-your-key-here
```
Or for Anthropic:
```bash
set ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### Step 2: Update config.py
Open `config.py` and set the provider:
```python
LLM_PROVIDER = "openai"  # or "anthropic"
```

### Step 3: Verify Setup
Run the system and check logs. You should see:
```
INFO: product_info: Querying OpenAI for product: ...
```
or
```
INFO: product_info: Querying Anthropic Claude for product: ...
```

## Testing LLM Integration

1. **Test with barcode not in database**:
   - Scan a barcode that's not in Open Food Facts
   - System should use LLM to get product info
   - Check logs for "using LLM fallback"

2. **Test with low confidence detection**:
   - Point camera at a product
   - If confidence < 60%, LLM should be used
   - Check logs for confidence percentage

3. **Test with out-of-dataset product**:
   - Point camera at a product not in training dataset
   - System should detect it's not in dataset
   - LLM should be used to get product info

## Troubleshooting

### LLM not being used
- Check if API key is set correctly
- Verify `LLM_PROVIDER` in config.py matches your API key
- Check logs for errors

### Incorrect product information
- Try a different LLM provider
- Check if the prompt is clear in logs
- Verify API key has sufficient credits

### API errors
- Check internet connection
- Verify API key is valid
- Check API key has sufficient credits/quota
- Try the other LLM provider

## Cost Considerations

### OpenAI GPT-4o-mini
- Very low cost (~$0.15 per 1M input tokens)
- Good for frequent use
- Recommended for this project

### Anthropic Claude 3.5 Sonnet
- Higher cost but more accurate
- Good for critical applications
- Better at understanding context

## Recommendations

For this grocery detection system:
- **Use OpenAI GPT-4o-mini** if you want low cost and good accuracy
- **Use Anthropic Claude 3.5 Sonnet** if you want maximum accuracy

Both are configured with:
- Low temperature (0.2) for consistency
- JSON response format
- Optimized prompts for grocery products

## No API Key?

If you don't have an API key, the system will still work but:
- Will use fallback descriptions (less accurate)
- Won't provide detailed product information
- May not work well for products not in database/dataset

Consider getting a free trial API key for better results!

