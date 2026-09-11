// server.js - OpenAI to NVIDIA NIM API Proxy (with Cloudflare Workers AI fallback)
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '128mb' }));

// NVIDIA NIM API configuration (primary provider)
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// Cloudflare Workers AI configuration (free fallback - used when NIM fails/rate-limits)
const CLOUDFLARE_ACCOUNT_ID = process.env.CLOUDFLARE_ACCOUNT_ID;
const CLOUDFLARE_API_TOKEN = process.env.CLOUDFLARE_API_TOKEN;
const CLOUDFLARE_API_BASE = CLOUDFLARE_ACCOUNT_ID
  ? `https://api.cloudflare.com/client/v4/accounts/${CLOUDFLARE_ACCOUNT_ID}/ai/v1`
  : null;

// 🔥 REASONING DISPLAY TOGGLE
const SHOW_REASONING = false;

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

// NIM model ID -> Cloudflare Workers AI equivalent. NOT identical models - Cloudflare
// doesn't host GLM-5.2/MiniMax/DeepSeek-V4 specifically, so these are the closest
// available substitutes. Quality/style WILL differ from the original model. Models with
// no reasonable Cloudflare equivalent are left unmapped - fallback is skipped for those.
const CLOUDFLARE_FALLBACK_MAPPING = {
  'z-ai/glm-5.2': '@cf/z-ai/glm-4.7-flash',
  'moonshotai/kimi-k3': '@cf/moonshotai/kimi-k2.5'
  // minimax-m3, deepseek-v4-flash/pro, step-3.7-flash: no Cloudflare equivalent, no entry here
};

const THINKING_MODELS = new Set([]);

const DEFAULT_MAX_TOKENS = 64000;
const MODEL_MAX_TOKENS = {
  'z-ai/glm-5.2': 64000,
  'deepseek-ai/deepseek-v4-flash-0731': 32000,
  'deepseek-ai/deepseek-v4-pro-0813': 32000
};

// Tries NIM first (with retries on 429). If NIM exhausts retries or errors out, and a
// Cloudflare equivalent exists, falls back to that - completely free either way.
async function postWithFallback(nimModel, nimRequestBody, axiosConfig, maxRetries = 3) {
  // --- Try NIM first ---
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequestBody, {
        ...axiosConfig,
        headers: { 'Authorization': `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' }
      });
      return { response, provider: 'nim', modelUsed: nimModel };
    } catch (err) {
      const status = err.response?.status;
      if (status === 429 && attempt < maxRetries) {
        const waitMs = 2000 * (attempt + 1);
        console.log(`[RETRY] NIM 429, retrying in ${waitMs}ms (attempt ${attempt + 1}/${maxRetries})`);
        await new Promise(resolve => setTimeout(resolve, waitMs));
        continue;
      }
      console.log(`[FALLBACK] NIM failed (status=${status || 'no response'}), checking Cloudflare fallback...`);
      break;
    }
  }

  // --- Fallback to Cloudflare Workers AI ---
  const cfModel = CLOUDFLARE_FALLBACK_MAPPING[nimModel];
  if (!cfModel || !CLOUDFLARE_API_BASE || !CLOUDFLARE_API_TOKEN) {
    console.log('[FALLBACK] No Cloudflare equivalent/credentials available, giving up.');
    throw new Error('NIM failed and no Cloudflare fallback is available for this model.');
  }

  console.log(`[FALLBACK] Trying Cloudflare with model=${cfModel}`);
  const cfBody = { ...nimRequestBody, model: cfModel };
  delete cfBody.chat_template_kwargs; // Cloudflare doesn't use NIM's thinking-mode param shape

  const response = await axios.post(`${CLOUDFLARE_API_BASE}/chat/completions`, cfBody, {
    ...axiosConfig,
    headers: {
      'Authorization': `Bearer ${CLOUDFLARE_API_TOKEN}`,
      'Content-Type': 'application/json'
    }
  });
  return { response, provider: 'cloudflare', modelUsed: cfModel };
}

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy (with Cloudflare fallback)',
    reasoning_display: SHOW_REASONING,
    thinking_models: Array.from(THINKING_MODELS),
    cloudflare_fallback_enabled: !!(CLOUDFLARE_API_BASE && CLOUDFLARE_API_TOKEN),
    cloudflare_fallback_models: Object.keys(CLOUDFLARE_FALLBACK_MAPPING)
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
  res.json({ object: 'list', data: models });
});

// Chat completions endpoint (main proxy)
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;
    console.log(`[REQUEST] model=${model} requested_max_tokens=${max_tokens} temperature=${temperature} stream=${stream}`);

    let nimModel = MODEL_MAPPING[model];
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

    const needsThinking = THINKING_MODELS.has(nimModel);

    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature || 0.75,
      max_tokens: max_tokens || MODEL_MAX_TOKENS[nimModel] || DEFAULT_MAX_TOKENS,
      chat_template_kwargs: needsThinking ? { thinking: true, enable_thinking: true } : undefined,
      stream: stream || false
    };

    const { response, provider, modelUsed } = await postWithFallback(nimModel, nimRequest, {
      responseType: stream ? 'stream' : 'json',
      timeout: 600000
    });
    console.log(`[PROVIDER USED] ${provider} (${modelUsed})`);

    if (stream) {
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
                  data.choices[0].delta.content = content || '';
                  delete data.choices[0].delta.reasoning_content;
                }
              }
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            } catch (e) {
              console.error('Skipped unparsable chunk:', line.slice(0, 200));
            }
          }
        });
      });

      response.data.on('end', () => {
        if (buffer.trim()) {
          res.write(buffer.startsWith('data: ') ? buffer + '\n\n' : buffer);
        }
        res.end();
      });
      response.data.on('error', (err) => {
        console.error('Stream error:', err);
        res.end();
      });
    } else {
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
            message: { role: choice.message.role, content: fullContent },
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
      };
      res.json(openaiResponse);
    }
  } catch (error) {
    let errorDetail = error.message;
    if (error.response?.data && typeof error.response.data.on === 'function') {
      try {
        errorDetail = await new Promise((resolve) => {
          let raw = '';
          error.response.data.on('data', (chunk) => { raw += chunk.toString(); });
          error.response.data.on('end', () => {
            try { resolve(JSON.parse(raw)); } catch { resolve(raw || error.message); }
          });
          error.response.data.on('error', () => resolve(error.message));
        });
      } catch { errorDetail = error.message; }
    } else if (error.response?.data) {
      errorDetail = error.response.data;
    }

    console.error('Proxy error:', errorDetail);

    let errorMessage;
    if (typeof errorDetail === 'string') errorMessage = errorDetail;
    else if (errorDetail?.error?.message) errorMessage = errorDetail.error.message;
    else if (errorDetail?.message) errorMessage = errorDetail.message;
    else if (errorDetail) {
      try { errorMessage = JSON.stringify(errorDetail); } catch { errorMessage = 'Internal server error'; }
    } else errorMessage = 'Internal server error';

    res.status(error.response?.status || 500).json({
      error: { message: errorMessage, type: 'invalid_request_error', code: error.response?.status || 500 }
    });
  }
});

// Catch-all for unsupported endpoints (Express 5-safe - see earlier note on app.use vs app.all('*'))
app.use((req, res) => {
  res.status(404).json({
    error: { message: `Endpoint ${req.path} not found`, type: 'invalid_request_error', code: 404 }
  });
});

app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`Cloudflare fallback: ${(CLOUDFLARE_API_BASE && CLOUDFLARE_API_TOKEN) ? 'ENABLED' : 'DISABLED (missing credentials)'}`);
});
