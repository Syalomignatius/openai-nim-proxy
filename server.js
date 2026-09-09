// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '128mb' }));

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// 🔥 REASONING DISPLAY TOGGLE - Shows/hides reasoning in output
const SHOW_REASONING = false; // Set to true to show reasoning with <think> tags

// 🔥 THINKING MODE - now PER-MODEL instead of global (see THINKING_MODELS below).
// Only models in that set get chat_template_kwargs attached; everything else is sent as a plain request.

// Model mapping (adjust based on available NIM models)
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4': 'qwen/qwen3-coder-480b-a35b-instruct',
  'gpt-oss-120b': 'openai/gpt-oss-120b',
  'kimi-k3': 'moonshotai/kimi-k3',
  'deepseek-v4-flash': 'deepseek-ai/deepseek-v4-flash-0731',
  'deepseek-v4-pro': 'deepseek-ai/deepseek-v4-pro-0813',
  'minimax-m3': 'minimaxai/minimax-m3',
  'step-3.7-flash': 'stepfun-ai/step-3.7-flash',
  'glm-5.2': 'z-ai/glm-5.2'
};

// Models (by their actual NIM model ID) that require chat_template_kwargs thinking flags.
// Left empty - DeepSeek Flash/Pro thinking mode disabled for Chub reliability.
// Add a model ID back here if you want to re-enable thinking mode for it.
const THINKING_MODELS = new Set([]);

// Per-model max_tokens fallback (used only when Chub doesn't send its own max_tokens).
// Anything not listed here falls back to DEFAULT_MAX_TOKENS.
const DEFAULT_MAX_TOKENS = 64000;
const MODEL_MAX_TOKENS = {
  'z-ai/glm-5.2': 64000,
  'deepseek-ai/deepseek-v4-flash-0731': 32000,
  'deepseek-ai/deepseek-v4-pro-0813': 32000
};

// Simple retry helper for 429s - waits and retries instead of failing immediately.
// Also logs the actual rate-limit headers NIM sends back (NIM doesn't expose useful ones,
// but kept here in case that changes).
async function postWithRetry(url, data, config, maxRetries = 3) {
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await axios.post(url, data, config);
    } catch (err) {
      const status = err.response?.status;
      if (status === 429) {
        const headers = err.response?.headers || {};
        console.log('[429 HEADERS]', {
          'retry-after': headers['retry-after'],
          'x-ratelimit-limit': headers['x-ratelimit-limit'],
          'x-ratelimit-remaining': headers['x-ratelimit-remaining'],
          'x-ratelimit-reset': headers['x-ratelimit-reset'],
          'all-headers': headers
        });

        if (attempt < maxRetries) {
          const waitMs = 2000 * (attempt + 1); // 2s, 4s, 6s backoff
          console.log(`[RETRY] Got 429, retrying in ${waitMs}ms (attempt ${attempt + 1}/${maxRetries})`);
          await new Promise(resolve => setTimeout(resolve, waitMs));
          continue;
        }
      }
      throw err;
    }
  }
}

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    service: 'OpenAI to NVIDIA NIM Proxy', 
    reasoning_display: SHOW_REASONING,
    thinking_models: Array.from(THINKING_MODELS)
  });
});

// List models endpoint (OpenAI compatible)
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));
  
  res.json({
    object: 'list',
    data: models
  });
});

// Chat completions endpoint (main proxy)
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;
    console.log(`[REQUEST] model=${model} requested_max_tokens=${max_tokens} temperature=${temperature} stream=${stream}`);
    
    // Smart model selection with fallback
    let nimModel = MODEL_MAPPING[model];
    if (!nimModel) {
      try {
        await axios.post(`${NIM_API_BASE}/chat/completions`, {
          model: model,
          messages: [{ role: 'user', content: 'test' }],
          max_tokens: 1
        }, {
          headers: { 'Authorization': `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
          validateStatus: (status) => status < 500
        }).then(res => {
          if (res.status >= 200 && res.status < 300) {
            nimModel = model;
          }
        });
      } catch (e) {}
      
      if (!nimModel) {
        const modelLower = model.toLowerCase();
        if (modelLower.includes('gpt-4') || modelLower.includes('claude-opus') || modelLower.includes('405b')) {
          nimModel = 'meta/llama-3.1-405b-instruct';
        } else if (modelLower.includes('claude') || modelLower.includes('gemini') || modelLower.includes('70b')) {
          nimModel = 'meta/llama-3.1-70b-instruct';
        } else {
          nimModel = 'meta/llama-3.1-8b-instruct';
        }
      }
    }
    
    // Only attach thinking params for models that actually need them
    const needsThinking = THINKING_MODELS.has(nimModel);

    // Transform OpenAI request to NIM format
    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature || 0.75,
      max_tokens: max_tokens || MODEL_MAX_TOKENS[nimModel] || DEFAULT_MAX_TOKENS,
      // NOTE: NIM requires chat_template_kwargs at the ROOT of the payload (not nested under extra_body),
      // and DeepSeek V4 reasoning models specifically require BOTH thinking + enable_thinking to be set.
      chat_template_kwargs: needsThinking ? { thinking: true, enable_thinking: true } : undefined,
      stream: stream || false
    };
    
    // Make request to NVIDIA NIM API (with automatic retry on 429 rate limits)
    const response = await postWithRetry(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json',
      timeout: 600000 // 600s (10min) - large models can take a while to fully respond
    });
    
    if (stream) {
      // Handle streaming response with reasoning
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      let buffer = '';
      let reasoningStarted = false;
      
      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        
        lines.forEach(line => {
          if (line.startsWith('data: ')) {
            if (line.includes('[DONE]')) {
              res.write(line + '\n');
              return;
            }
            
            try {
              const data = JSON.parse(line.slice(6));
              if (data.choices?.[0]?.finish_reason) {
                console.log(`[STREAM END] finish_reason=${data.choices[0].finish_reason}`);
              }
              if (data.choices?.[0]?.delta) {
                const reasoning = data.choices[0].delta.reasoning_content;
                const content = data.choices[0].delta.content;
                
                if (SHOW_REASONING) {
                  let combinedContent = '';
                  
                  if (reasoning && !reasoningStarted) {
                    combinedContent = '<think>\n' + reasoning;
                    reasoningStarted = true;
                  } else if (reasoning) {
                    combinedContent = reasoning;
                  }
                  
                  if (content && reasoningStarted) {
                    combinedContent += '</think>\n\n' + content;
                    reasoningStarted = false;
                  } else if (content) {
                    combinedContent += content;
                  }
                  
                  if (combinedContent) {
                    data.choices[0].delta.content = combinedContent;
                    delete data.choices[0].delta.reasoning_content;
                  }
                } else {
                  if (content) {
                    data.choices[0].delta.content = content;
                  } else {
                    data.choices[0].delta.content = '';
                  }
                  delete data.choices[0].delta.reasoning_content;
                }
              }
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            } catch (e) {
              // Don't forward broken/unparsable chunks - passing malformed JSON downstream
              // makes Chub's own parser crash trying to read .delta off it. Skip and log instead.
              console.error('Skipped unparsable NIM chunk:', line.slice(0, 200));
            }
          }
        });
      });
      
      response.data.on('end', () => {
        // Flush any leftover partial line still sitting in the buffer -
        // without this, the final chunk (sometimes the last content delta or [DONE]) gets silently dropped.
        if (buffer.trim()) {
          if (buffer.startsWith('data: ')) {
            res.write(buffer + '\n\n');
          } else {
            res.write(buffer);
          }
        }
        res.end();
      });
      response.data.on('error', (err) => {
        console.error('Stream error:', err);
        res.end();
      });
    } else {
      // Transform NIM response to OpenAI format with reasoning
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices.map(choice => {
          let fullContent = choice.message?.content || '';
          
          if (SHOW_REASONING && choice.message?.reasoning_content) {
            fullContent = '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' + fullContent;
          }
          
          return {
            index: choice.index,
            message: {
              role: choice.message.role,
              content: fullContent
            },
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };
      
      res.json(openaiResponse);
    }
    
  } catch (error) {
    // If the request was made with responseType: 'stream', error.response.data is a raw
    // unread stream, not parsed JSON - dumping it directly logs a massive internal object.
    // Read it properly to get NIM's actual error message.
    let errorDetail = error.message;
    if (error.response?.data && typeof error.response.data.on === 'function') {
      try {
        errorDetail = await new Promise((resolve) => {
          let raw = '';
          error.response.data.on('data', (chunk) => { raw += chunk.toString(); });
          error.response.data.on('end', () => {
            try {
              resolve(JSON.parse(raw));
            } catch {
              resolve(raw || error.message);
            }
          });
          error.response.data.on('error', () => resolve(error.message));
        });
      } catch {
        errorDetail = error.message;
      }
    } else if (error.response?.data) {
      errorDetail = error.response.data;
    }

    console.error('Proxy error:', errorDetail);
    
    // Always resolve to a plain string - never let a raw object leak into the message field,
    // otherwise clients display it as the literal text "[object Object]".
    let errorMessage;
    if (typeof errorDetail === 'string') {
      errorMessage = errorDetail;
    } else if (errorDetail?.error?.message) {
      errorMessage = errorDetail.error.message;
    } else if (errorDetail?.message) {
      errorMessage = errorDetail.message;
    } else if (errorDetail) {
      try {
        errorMessage = JSON.stringify(errorDetail);
      } catch {
        errorMessage = 'Internal server error';
      }
    } else {
      errorMessage = 'Internal server error';
    }

    res.status(error.response?.status || 500).json({
      error: {
        message: errorMessage,
        type: 'invalid_request_error',
        code: error.response?.status || 500
      }
    });
  }
});

// Catch-all for unsupported endpoints
// NOTE: Express 5 changed how bare '*' wildcards work in routes (path-to-regexp v7 requires
// named wildcards like '*splat'). Using app.use() with no path instead sidesteps that entirely -
// it matches all methods and paths the same way app.all('*', ...) did in Express 4.
app.use((req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Thinking mode enabled for: ${Array.from(THINKING_MODELS).join(', ') || 'none'}`);
});
