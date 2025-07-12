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
  Tool,
  FunctionCall,
} from '@google/genai';
import { ContentGenerator } from './contentGenerator.js';
import { Config } from '../config/config.js';
import {
  logApiRequest,
  logApiResponse,
  logApiError,
} from '../telemetry/loggers.js';
import {
  ApiErrorEvent,
  ApiRequestEvent,
  ApiResponseEvent,
} from '../telemetry/types.js';

interface OpenAIMessage {
  role: 'system' | 'user' | 'assistant';
  content: string | null;
  tool_calls?: OpenAIToolCall[];
}

interface OpenAIToolCall {
  id: string;
  type: 'function';
  function: {
    name: string;
    arguments: string;
  };
}

interface OpenAITool {
  type: 'function';
  function: {
    name: string;
    description?: string;
    parameters?: Record<string, unknown>;
  };
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
      content: string | null;
      tool_calls?: OpenAIToolCall[];
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
      tool_calls?: OpenAIToolCall[];
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
  private readonly config?: Config;

  constructor(
    baseUrl: string,
    model: string,
    apiKey?: string,
    customHeaders?: Record<string, string>,
    config?: Config,
  ) {
    this.baseUrl = baseUrl.replace(/\/$/, ''); // Remove trailing slash
    this.apiKey = apiKey;
    this.model = model;
    this.config = config;
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

    // Create parts array from content and tool calls
    const parts: Part[] = [];

    // Add text content if present
    if (choice.message.content) {
      parts.push({ text: choice.message.content });
    }

    // Add function calls if present
    if (choice.message.tool_calls && choice.message.tool_calls.length > 0) {
      const functionCallParts = this.convertOpenAIToolCallsToGemini(choice.message.tool_calls);
      parts.push(...functionCallParts);
    }

    // Create a basic response that matches the GenerateContentResponse structure
    const response = new GenerateContentResponse();
    response.candidates = [{
      index: choice.index,
      content: {
        role: 'model',
        parts: parts.length > 0 ? parts : [{ text: '' }]
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

  /**
   * Convert Gemini tools to OpenAI tools format
   */
  private convertGeminiToolsToOpenAI(tools?: Tool[]): OpenAITool[] | undefined {
    if (!tools || !tools.length) {
      return undefined;
    }

    const openaiTools: OpenAITool[] = [];
    for (const tool of tools) {
      if (tool.functionDeclarations) {
        for (const funcDecl of tool.functionDeclarations) {
          if (funcDecl.name) {
            openaiTools.push({
              type: 'function',
              function: {
                name: funcDecl.name,
                description: funcDecl.description,
                parameters: funcDecl.parameters as Record<string, unknown> | undefined,
              },
            });
          }
        }
      }
    }
    return openaiTools.length > 0 ? openaiTools : undefined;
  }

  /**
   * Convert OpenAI tool calls to Gemini function call parts
   */
  private convertOpenAIToolCallsToGemini(toolCalls: OpenAIToolCall[]): Part[] {
    return toolCalls.map((toolCall) => {
      let args: Record<string, unknown> = {};
      try {
        args = JSON.parse(toolCall.function.arguments);
      } catch (e) {
        // If arguments aren't valid JSON, leave as empty object
      }

      const functionCall: FunctionCall = {
        name: toolCall.function.name,
        args,
        id: toolCall.id,
      };

      return { functionCall };
    });
  }

  /**
   * Extract text from Contents for logging purposes
   */
  private getRequestTextFromContents(contents: ContentListUnion): string {
    const normalizedContents = this.normalizeContents(contents);
    return normalizedContents
      .flatMap((content) => content.parts ?? [])
      .map((part) => part.text)
      .filter(Boolean)
      .join('');
  }

  /**
   * Log API request
   */
  private logApiRequest(
    contents: ContentListUnion,
    model: string,
    prompt_id: string,
  ): void {
    if (!this.config) return;
    const requestText = this.getRequestTextFromContents(contents);
    logApiRequest(
      this.config,
      new ApiRequestEvent(model, prompt_id, requestText),
    );
  }

  /**
   * Log API response
   */
  private logApiResponse(
    durationMs: number,
    prompt_id: string,
    model: string,
    usageTokens?: { prompt_tokens: number; completion_tokens: number; total_tokens: number },
    responseText?: string,
  ): void {
    if (!this.config) return;
    
    // Convert OpenAI usage format to Gemini format
    const usageMetadata = usageTokens ? {
      promptTokenCount: usageTokens.prompt_tokens,
      candidatesTokenCount: usageTokens.completion_tokens,
      totalTokenCount: usageTokens.total_tokens,
      cachedContentTokenCount: 0,
      thoughtsTokenCount: 0,
      toolTokenCount: 0,
    } : undefined;

    logApiResponse(
      this.config,
      new ApiResponseEvent(
        model,
        durationMs,
        prompt_id,
        usageMetadata,
        responseText,
      ),
    );
  }

  /**
   * Log API error
   */
  private logApiError(
    durationMs: number,
    prompt_id: string,
    model: string,
    error: Error,
    statusCode?: number,
  ): void {
    if (!this.config) return;
    logApiError(
      this.config,
      new ApiErrorEvent(
        model,
        error.message,
        durationMs,
        prompt_id,
        error.constructor.name,
        statusCode,
      ),
    );
  }

  async generateContent(request: GenerateContentParameters): Promise<GenerateContentResponse> {
    const startTime = Date.now();
    const modelToUse = request.model || this.model;
    const prompt_id = 'openai-generate-content'; // Fallback since we don't have access to prompt_id in this interface
    
    // Extract system instruction from config
    const systemInstruction = this.extractTextFromContentUnion(request.config?.systemInstruction);

    // Log the request
    this.logApiRequest(request.contents, modelToUse, prompt_id);

    const messages = this.convertToOpenAIMessages(request.contents, systemInstruction);

    // Convert Gemini tools to OpenAI format
    const requestTools = request.config?.tools;
    let tools: OpenAITool[] | undefined;
    if (requestTools && Array.isArray(requestTools)) {
      // Filter to only Tool objects (not CallableTool)
      const geminiTools = requestTools.filter((tool): tool is Tool =>
        tool && typeof tool === 'object' && 'functionDeclarations' in tool
      );
      tools = this.convertGeminiToolsToOpenAI(geminiTools);
    }

    const openaiRequest: any = {
      model: modelToUse,
      messages,
      max_tokens: request.config?.maxOutputTokens || 4096,
      temperature: request.config?.temperature || 0.7,
      top_p: request.config?.topP || 0.9,
      stream: false,
    };

    // Only add tools if they exist
    if (tools && tools.length > 0) {
      openaiRequest.tools = tools;
      openaiRequest.tool_choice = 'auto';
    }

    try {
      const response = await fetch(`${this.baseUrl}/v1/chat/completions`, {
        method: 'POST',
        headers: this.headers,
        body: JSON.stringify(openaiRequest),
        signal: request.config?.abortSignal,
      });

      if (!response.ok) {
        const durationMs = Date.now() - startTime;
        const error = new Error(`OpenAI API error: ${response.status} ${response.statusText}`);
        this.logApiError(durationMs, prompt_id, modelToUse, error, response.status);
        throw error;
      }

      const openaiResponse: OpenAIResponse = await response.json();
      const geminiResponse = this.convertToGeminiResponse(openaiResponse);
      
      // Log successful response
      const durationMs = Date.now() - startTime;
      const responseText = geminiResponse.candidates?.[0]?.content?.parts
        ?.map(part => part.text)
        .filter(Boolean)
        .join('') || '';
      
      this.logApiResponse(
        durationMs,
        prompt_id,
        modelToUse,
        openaiResponse.usage,
        responseText,
      );

      return geminiResponse;
    } catch (error) {
      const durationMs = Date.now() - startTime;
      if (error instanceof Error) {
        this.logApiError(durationMs, prompt_id, modelToUse, error);
      }
      throw error;
    }
  }

  async generateContentStream(
    request: GenerateContentParameters,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    const startTime = Date.now();
    const modelToUse = request.model || this.model;
    const prompt_id = 'openai-generate-stream'; // Fallback since we don't have access to prompt_id in this interface
    
    // Extract system instruction from config
    const systemInstruction = this.extractTextFromContentUnion(request.config?.systemInstruction);

    // Log the request
    this.logApiRequest(request.contents, modelToUse, prompt_id);

    const messages = this.convertToOpenAIMessages(request.contents, systemInstruction);

    // Convert Gemini tools to OpenAI format
    const requestTools = request.config?.tools;
    let tools: OpenAITool[] | undefined;
    if (requestTools && Array.isArray(requestTools)) {
      // Filter to only Tool objects (not CallableTool)
      const geminiTools = requestTools.filter((tool): tool is Tool =>
        tool && typeof tool === 'object' && 'functionDeclarations' in tool
      );
      tools = this.convertGeminiToolsToOpenAI(geminiTools);
    }

    const openaiRequest: any = {
      model: modelToUse,
      messages,
      max_tokens: request.config?.maxOutputTokens || 4096,
      temperature: request.config?.temperature || 0.7,
      top_p: request.config?.topP || 0.9,
      stream: true,
    };

    // Only add tools if they exist
    if (tools && tools.length > 0) {
      openaiRequest.tools = tools;
      openaiRequest.tool_choice = 'auto';
    }

    try {
      const response = await fetch(`${this.baseUrl}/v1/chat/completions`, {
        method: 'POST',
        headers: this.headers,
        body: JSON.stringify(openaiRequest),
        signal: request.config?.abortSignal,
      });

      if (!response.ok) {
        const durationMs = Date.now() - startTime;
        const error = new Error(`OpenAI API error: ${response.status} ${response.statusText}`);
        this.logApiError(durationMs, prompt_id, modelToUse, error, response.status);
        throw error;
      }

      // For streaming, we'll log the response when the stream completes
      return this.parseStreamResponse(response, startTime, prompt_id, modelToUse);
    } catch (error) {
      const durationMs = Date.now() - startTime;
      if (error instanceof Error) {
        this.logApiError(durationMs, prompt_id, modelToUse, error);
      }
      throw error;
    }
  }

  private async *parseStreamResponse(
    response: Response,
    startTime: number,
    prompt_id: string,
    modelToUse: string,
  ): AsyncGenerator<GenerateContentResponse> {
    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('No response body');
    }

    const decoder = new TextDecoder();
    let buffer = '';
    let totalResponseText = '';

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
              // Log the completed streaming response
              const durationMs = Date.now() - startTime;
              this.logApiResponse(
                durationMs,
                prompt_id,
                modelToUse,
                undefined, // OpenAI streaming doesn't provide usage in individual chunks
                totalResponseText,
              );
              return;
            }

            try {
              const parsed: OpenAIStreamResponse = JSON.parse(data);
              const choice = parsed.choices[0];
              if (choice?.delta) {
                const parts: Part[] = [];

                // Add text content if present
                if (choice.delta.content) {
                  parts.push({ text: choice.delta.content });
                  totalResponseText += choice.delta.content;
                }

                // Add function calls if present
                if (choice.delta.tool_calls && choice.delta.tool_calls.length > 0) {
                  const functionCallParts = this.convertOpenAIToolCallsToGemini(choice.delta.tool_calls);
                  parts.push(...functionCallParts);
                }

                // Only yield if we have content
                if (parts.length > 0) {
                  const geminiResponse = new GenerateContentResponse();
                  geminiResponse.candidates = [{
                    index: choice.index,
                    content: {
                      role: 'model',
                      parts
                    },
                    finishReason: choice.finish_reason === 'stop' ? FinishReason.STOP :
                      choice.finish_reason ? FinishReason.OTHER : undefined,
                    safetyRatings: []
                  }];
                  geminiResponse.modelVersion = parsed.model;

                  yield geminiResponse;
                }
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
