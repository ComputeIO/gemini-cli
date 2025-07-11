/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  CountTokensResponse,
  GenerateContentResponse,
  GenerateContentParameters,
  CountTokensParameters,
  EmbedContentResponse,
  EmbedContentParameters,
  Content,
  Part,
  ContentListUnion,
  FinishReason,
} from '@google/genai';
import { ContentGenerator } from './contentGenerator.js';

interface OpenAIMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

interface OpenAIResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    message: {
      role: string;
      content: string;
    };
    finish_reason: string;
  }>;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

interface OpenAIStreamResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    delta: {
      role?: string;
      content?: string;
    };
    finish_reason?: string;
  }>;
}

/**
 * OpenAI-compatible content generator that implements the ContentGenerator interface.
 * This allows the Gemini CLI to work with Ollama, OpenAI, and other OpenAI-compatible APIs.
 */
export class OpenAICompatibleGenerator implements ContentGenerator {
  private readonly baseUrl: string;
  private readonly apiKey?: string;
  private readonly model: string;
  private readonly headers: Record<string, string>;

  constructor(
    baseUrl: string,
    model: string,
    apiKey?: string,
    customHeaders?: Record<string, string>,
  ) {
    this.baseUrl = baseUrl.replace(/\/$/, ''); // Remove trailing slash
    this.apiKey = apiKey;
    this.model = model;
    this.headers = {
      'Content-Type': 'application/json',
      ...customHeaders,
    };

    if (this.apiKey) {
      this.headers['Authorization'] = `Bearer ${this.apiKey}`;
    }
  }

  /**
   * Normalize ContentListUnion to Content array
   */
  private normalizeContents(contents: ContentListUnion): Content[] {
    if (typeof contents === 'string') {
      return [{
        role: 'user',
        parts: [{ text: contents }]
      }];
    }
    
    if (Array.isArray(contents)) {
      // Check if it's an array of Content objects or Part objects
      if (contents.length > 0 && contents[0] && typeof contents[0] === 'object' && 'role' in contents[0]) {
        return contents as Content[];
      } else {
        // It's an array of parts, wrap in a user Content
        return [{
          role: 'user',
          parts: contents as Part[]
        }];
      }
    }
    
    // Single Content object
    if ('role' in contents) {
      return [contents as Content];
    }
    
    // Single Part object
    return [{
      role: 'user',
      parts: [contents as Part]
    }];
  }

  /**
   * Extract text from ContentUnion (which can be string, Content, Part, or arrays)
   */
  private extractTextFromContentUnion(content?: any): string | undefined {
    if (!content) {
      return undefined;
    }
    
    if (typeof content === 'string') {
      return content;
    }
    
    if (Array.isArray(content)) {
      // Array of parts
      return content
        .filter((part: any) => part && typeof part === 'object' && 'text' in part)
        .map((part: any) => part.text)
        .join('\n') || undefined;
    }
    
    if (typeof content === 'object' && 'parts' in content) {
      // Content object
      return (content.parts || [])
        .filter((part: any) => part && typeof part === 'object' && 'text' in part)
        .map((part: any) => part.text)
        .join('\n') || undefined;
    }
    
    if (typeof content === 'object' && 'text' in content) {
      // Single Part object
      return content.text;
    }
    
    return undefined;
  }

  /**
   * Convert Gemini Content format to OpenAI messages format
   */
  private convertToOpenAIMessages(contents: ContentListUnion, systemInstruction?: string): OpenAIMessage[] {
    const messages: OpenAIMessage[] = [];
    
    // Add system instruction as the first message if provided
    if (systemInstruction && systemInstruction.trim()) {
      messages.push({ role: 'system', content: systemInstruction.trim() });
    }
    
    const normalizedContents = this.normalizeContents(contents);

    for (const content of normalizedContents) {
      const role = content.role === 'user' ? 'user' : 'assistant';
      const text = (content.parts || [])
        .filter((part: Part) => part && typeof part === 'object' && 'text' in part)
        .map((part: Part) => (part as { text: string }).text)
        .join('\n');

      if (text.trim()) {
        messages.push({ role, content: text });
      }
    }

    return messages;
  }

  /**
   * Convert OpenAI response to Gemini format
   */
  private convertToGeminiResponse(openaiResponse: OpenAIResponse): GenerateContentResponse {
    const choice = openaiResponse.choices[0];
    if (!choice) {
      throw new Error('No choices in OpenAI response');
    }

    // Create a basic response that matches the GenerateContentResponse structure
    const response = new GenerateContentResponse();
    response.candidates = [{
      index: choice.index,
      content: {
        role: 'model',
        parts: [{ text: choice.message.content }]
      },
      finishReason: choice.finish_reason === 'stop' ? FinishReason.STOP : FinishReason.OTHER,
      safetyRatings: []
    }];

    response.usageMetadata = {
      promptTokenCount: openaiResponse.usage.prompt_tokens,
      candidatesTokenCount: openaiResponse.usage.completion_tokens,
      totalTokenCount: openaiResponse.usage.total_tokens
    };

    response.modelVersion = openaiResponse.model;

    return response;
  }

  async generateContent(request: GenerateContentParameters): Promise<GenerateContentResponse> {
    // Extract system instruction from config
    const systemInstruction = this.extractTextFromContentUnion(request.config?.systemInstruction);
    
    const messages = this.convertToOpenAIMessages(request.contents, systemInstruction);

    const openaiRequest = {
      model: request.model || this.model,
      messages,
      max_tokens: request.config?.maxOutputTokens || 4096,
      temperature: request.config?.temperature || 0.7,
      top_p: request.config?.topP || 0.9,
      stream: false,
    };

    const response = await fetch(`${this.baseUrl}/v1/chat/completions`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify(openaiRequest),
      signal: request.config?.abortSignal,
    });

    if (!response.ok) {
      throw new Error(`OpenAI API error: ${response.status} ${response.statusText}`);
    }

    const openaiResponse: OpenAIResponse = await response.json();
    return this.convertToGeminiResponse(openaiResponse);
  }

  async generateContentStream(
    request: GenerateContentParameters,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    // Extract system instruction from config
    const systemInstruction = this.extractTextFromContentUnion(request.config?.systemInstruction);
    
    const messages = this.convertToOpenAIMessages(request.contents, systemInstruction);

    const openaiRequest = {
      model: request.model || this.model,
      messages,
      max_tokens: request.config?.maxOutputTokens || 4096,
      temperature: request.config?.temperature || 0.7,
      top_p: request.config?.topP || 0.9,
      stream: true,
    };

    const response = await fetch(`${this.baseUrl}/v1/chat/completions`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify(openaiRequest),
      signal: request.config?.abortSignal,
    });

    if (!response.ok) {
      throw new Error(`OpenAI API error: ${response.status} ${response.statusText}`);
    }

    return this.parseStreamResponse(response);
  }

  private async *parseStreamResponse(response: Response): AsyncGenerator<GenerateContentResponse> {
    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('No response body');
    }

    const decoder = new TextDecoder();
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (trimmed.startsWith('data: ')) {
            const data = trimmed.slice(6);
            if (data === '[DONE]') {
              return;
            }

            try {
              const parsed: OpenAIStreamResponse = JSON.parse(data);
              const choice = parsed.choices[0];
              if (choice?.delta?.content) {
                const geminiResponse = new GenerateContentResponse();
                geminiResponse.candidates = [{
                  index: choice.index,
                  content: {
                    role: 'model',
                    parts: [{ text: choice.delta.content }]
                  },
                  finishReason: choice.finish_reason === 'stop' ? FinishReason.STOP : 
                               choice.finish_reason ? FinishReason.OTHER : undefined,
                  safetyRatings: []
                }];
                geminiResponse.modelVersion = parsed.model;

                yield geminiResponse;
              }
            } catch (e) {
              // Skip invalid JSON
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }

  async countTokens(request: CountTokensParameters): Promise<CountTokensResponse> {
    // Extract system instruction from config if available
    const systemInstruction = this.extractTextFromContentUnion(request.config?.systemInstruction);
    
    const messages = this.convertToOpenAIMessages(request.contents, systemInstruction);

    // For OpenAI-compatible APIs, we estimate tokens based on character count
    // This is a rough approximation: ~4 characters per token for English text
    const totalText = messages.map(m => m.content).join(' ');
    const estimatedTokens = Math.ceil(totalText.length / 4);

    return {
      totalTokens: estimatedTokens,
    };
  }

  async embedContent(request: EmbedContentParameters): Promise<EmbedContentResponse> {
    // Extract text from contents
    const normalizedContents = this.normalizeContents(request.contents);
    const text = normalizedContents
      .flatMap(content => content.parts || [])
      .filter((part: Part) => part && typeof part === 'object' && 'text' in part)
      .map((part: Part) => (part as { text: string }).text)
      .join('\n');

    const embeddingRequest = {
      model: request.model || this.model,
      input: text,
    };

    const response = await fetch(`${this.baseUrl}/v1/embeddings`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify(embeddingRequest),
    });

    if (!response.ok) {
      throw new Error(`OpenAI API error: ${response.status} ${response.statusText}`);
    }

    const embeddingResponse = await response.json();
    const embedding = embeddingResponse.data?.[0]?.embedding;

    if (!embedding) {
      throw new Error('No embedding returned from API');
    }

    return {
      embeddings: [{
        values: embedding,
      }],
    };
  }
}
