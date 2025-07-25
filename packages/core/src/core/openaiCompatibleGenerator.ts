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
  ContentUnion,
  PartUnion,
} from '@google/genai';
import { AuthType, ContentGenerator } from './contentGenerator.js';
import { Config } from '../config/config.js';
import { logApiError } from '../telemetry/loggers.js';
import { ApiErrorEvent } from '../telemetry/types.js';

interface OpenAIMessage {
  role: 'system' | 'user' | 'assistant';
  content: string | null;
  tool_calls?: OpenAIToolCall[];
}

interface OpenAIToolCall {
  id?: string; // Made optional since it might not be present in streaming chunks
  type: 'function';
  index?: number; // Present in streaming responses
  function: {
    name?: string; // Made optional since it might come in pieces
    arguments?: string; // Made optional since it might come in pieces
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

  private normalizeBody(body: string): string {
    let n = body;
    if (this.baseUrl.startsWith('https://api.deepseek.com')) {
      n = n.replace(
        /"type":"(TYPE_UNSPECIFIED|STRING|NUMBER|INTEGER|BOOLEAN|ARRAY|OBJECT|NULL)"/g,
        (m) => m.toLowerCase(),
      );
      n = n.replace(/,"minLength":"\d+"/g, ''); // Remove minLength for compatibility
      n = n.replace(
        /,"minItems":"\d+"/g,
        (m) => `,"minItems":${m.match(/\d+/)?.[0] || 0}`,
      ); // Convert minItems to number
    }
    return n;
  }

  /**
   * Normalize ContentListUnion to Content array
   */
  private normalizeContents(contents: ContentListUnion): Content[] {
    if (typeof contents === 'string') {
      return [
        {
          role: 'user',
          parts: [{ text: contents }],
        },
      ];
    }

    if (Array.isArray(contents)) {
      // Check if it's an array of Content objects or Part objects
      if (
        contents.length > 0 &&
        contents[0] &&
        typeof contents[0] === 'object' &&
        'role' in contents[0]
      ) {
        return contents as Content[];
      } else {
        // It's an array of parts, wrap in a user Content
        return [
          {
            role: 'user',
            parts: contents as Part[],
          },
        ];
      }
    }

    // Single Content object
    if ('role' in contents) {
      return [contents as Content];
    }

    // Single Part object
    return [
      {
        role: 'user',
        parts: [contents as Part],
      },
    ];
  }

  /**
   * Extract text from ContentUnion (which can be string, Content, Part, or arrays)
   */
  private extractTextFromContentUnion(
    content?: ContentUnion,
  ): string | undefined {
    if (!content) {
      return undefined;
    }

    if (typeof content === 'string') {
      return content;
    }

    if (Array.isArray(content)) {
      // Array of parts
      return (
        content
          .filter((part) => part && typeof part === 'object' && 'text' in part)
          .map((part: PartUnion) =>
            typeof part === 'object' && 'text' in part ? part.text : part,
          )
          .join('\n') || undefined
      );
    }

    if (typeof content === 'object' && 'parts' in content) {
      // Content object
      return (
        (content.parts || [])
          .filter((part) => part && typeof part === 'object' && 'text' in part)
          .map((part: PartUnion) =>
            typeof part === 'object' && 'text' in part ? part.text : part,
          )
          .join('\n') || undefined
      );
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
  private convertToOpenAIMessages(
    contents: ContentListUnion,
    systemInstruction?: string,
  ): OpenAIMessage[] {
    const messages: OpenAIMessage[] = [];

    // Add system instruction as the first message if provided
    if (systemInstruction && systemInstruction.trim()) {
      messages.push({ role: 'system', content: systemInstruction.trim() });
    }

    const normalizedContents = this.normalizeContents(contents);

    for (const content of normalizedContents) {
      const role = content.role === 'user' ? 'user' : 'assistant';
      const text = (content.parts || [])
        .filter(
          (part: Part) => part && typeof part === 'object' && 'text' in part,
        )
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
  private convertToGeminiResponse(
    openaiResponse: OpenAIResponse,
  ): GenerateContentResponse {
    const choice = openaiResponse.choices[0];
    if (!choice) {
      throw new Error('No choices in OpenAI response');
    }

    // Create parts array from content and tool calls
    const parts: Part[] = [];

    // Add text content if present
    if (choice.message.content) {
      parts.push({ text: this.trimThink(choice.message.content) });
    }

    // Extract function calls for both parts and root-level functionCalls
    let functionCalls: FunctionCall[] = [];
    if (choice.message.tool_calls && choice.message.tool_calls.length > 0) {
      // Convert OpenAI tool calls to Gemini function calls
      functionCalls = choice.message.tool_calls.map((toolCall) => {
        let args: Record<string, unknown> = {};
        try {
          if (toolCall.function.arguments) {
            args = JSON.parse(toolCall.function.arguments);
          }
        } catch (_e) {
          // If arguments aren't valid JSON, leave as empty object
        }

        return {
          name: toolCall.function.name || 'unknown_function',
          args,
          id: toolCall.id || `tool_${toolCall.index || 0}`,
        } as FunctionCall;
      });

      // Also add function call parts for compatibility
      const functionCallParts = this.convertOpenAIToolCallsToGemini(
        choice.message.tool_calls,
      );
      parts.push(...functionCallParts);
    }

    // Create a response object that matches the GenerateContentResponse structure
    const responseObj = {
      candidates: [
        {
          index: choice.index,
          content: {
            role: 'model',
            parts: parts.length > 0 ? parts : [{ text: '' }],
          },
          finishReason:
            choice.finish_reason === 'stop'
              ? FinishReason.STOP
              : FinishReason.OTHER,
          safetyRatings: [],
        },
      ],
      usageMetadata: {
        promptTokenCount: openaiResponse.usage.prompt_tokens,
        candidatesTokenCount: openaiResponse.usage.completion_tokens,
        totalTokenCount: openaiResponse.usage.total_tokens,
      },
      modelVersion: openaiResponse.model,
      functionCalls: functionCalls,
    } as unknown as GenerateContentResponse;

    return responseObj;
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
                parameters: funcDecl.parameters as
                  | Record<string, unknown>
                  | undefined,
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
        if (toolCall.function.arguments) {
          args = JSON.parse(toolCall.function.arguments);
        }
      } catch (_e) {
        // If arguments aren't valid JSON, leave as empty object
      }

      const functionCall: FunctionCall = {
        name: toolCall.function.name || 'unknown_function',
        args,
        id: toolCall.id || `tool_${toolCall.index || 0}`,
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
        AuthType.USE_CUSTOM_OPENAI_COMPATIBLE,
        error.constructor.name,
        statusCode,
      ),
    );
  }

  /**
   * Extract headers from Response object for logging
   */
  private extractResponseHeaders(response: Response): Record<string, string> {
    const headers: Record<string, string> = {};
    if (response.headers && typeof response.headers.forEach === 'function') {
      response.headers.forEach((value, key) => {
        headers[key] = value;
      });
    }
    return headers;
  }

  async generateContent(
    request: GenerateContentParameters,
  ): Promise<GenerateContentResponse> {
    const startTime = Date.now();
    const modelToUse = request.model || this.model;
    const prompt_id = 'openai-generate-content'; // Fallback since we don't have access to prompt_id in this interface

    // Extract system instruction from config
    // const systemInstruction = this.extractTextFromContentUnion(request.config?.systemInstruction);
    const messages = this.convertToOpenAIMessages(request.contents); // function call instead of systemInstruction

    // Convert Gemini tools to OpenAI format
    const requestTools = request.config?.tools;
    let tools: OpenAITool[] | undefined;
    if (requestTools && Array.isArray(requestTools)) {
      // Filter to only Tool objects (not CallableTool)
      const geminiTools = requestTools.filter(
        (tool): tool is Tool =>
          tool && typeof tool === 'object' && 'functionDeclarations' in tool,
      );
      tools = this.convertGeminiToolsToOpenAI(geminiTools);
    }

    const openaiRequest = {
      model: modelToUse,
      messages,
      max_tokens: request.config?.maxOutputTokens || 4096,
      temperature: request.config?.temperature || 0.7,
      top_p: request.config?.topP || 0.9,
      stream: false,
      tools: tools && tools.length > 0 ? tools : undefined,
      tools_choice: tools && tools.length > 0 ? 'auto' : undefined,
    };

    const requestUrl = `${this.baseUrl}/v1/chat/completions`;

    try {
      const response = await fetch(requestUrl, {
        method: 'POST',
        headers: this.headers,
        body: this.normalizeBody(JSON.stringify(openaiRequest)),
        signal: request.config?.abortSignal,
      });

      if (!response.ok) {
        const durationMs = Date.now() - startTime;
        const responseHeaders = this.extractResponseHeaders(response);
        const error = new Error(
          `OpenAI API error: ${response.status} ${response.statusText}`,
        );

        // Log error with response details
        if (this.config?.getDebugMode()) {
          console.log(`[DEBUG] OpenAI API Error Response (${prompt_id}):`);
          console.log(`  Status: ${response.status} ${response.statusText}`);
          console.log(
            `  Response Headers:`,
            JSON.stringify(responseHeaders, null, 2),
          );
        }

        this.logApiError(
          durationMs,
          prompt_id,
          modelToUse,
          error,
          response.status,
        );
        throw error;
      }

      const openaiResponse: OpenAIResponse = await response.json();
      const geminiResponse = this.convertToGeminiResponse(openaiResponse);

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
    // const systemInstruction = this.extractTextFromContentUnion(request.config?.systemInstruction);
    const messages = this.convertToOpenAIMessages(request.contents); // function call instead of systemInstruction

    // Convert Gemini tools to OpenAI format
    const requestTools = request.config?.tools;
    let tools: OpenAITool[] | undefined;
    if (requestTools && Array.isArray(requestTools)) {
      // Filter to only Tool objects (not CallableTool)
      const geminiTools = requestTools.filter(
        (tool): tool is Tool =>
          tool && typeof tool === 'object' && 'functionDeclarations' in tool,
      );
      tools = this.convertGeminiToolsToOpenAI(geminiTools);
    }

    const openaiRequest = {
      model: modelToUse,
      messages,
      max_tokens: request.config?.maxOutputTokens || 4096,
      temperature: request.config?.temperature || 0.7,
      top_p: request.config?.topP || 0.9,
      stream: true,
      tools: undefined as OpenAITool[] | undefined,
      tool_choice: undefined as string | undefined,
    };

    // Only add tools if they exist
    if (tools && tools.length > 0) {
      openaiRequest.tools = tools;
      openaiRequest.tool_choice = 'auto';
    }

    // Log the request
    const requestUrl = `${this.baseUrl}/v1/chat/completions`;

    try {
      const response = await fetch(requestUrl, {
        method: 'POST',
        headers: this.headers,
        body: this.normalizeBody(JSON.stringify(openaiRequest)),
        signal: request.config?.abortSignal,
      });

      if (!response.ok) {
        const durationMs = Date.now() - startTime;
        const responseHeaders = this.extractResponseHeaders(response);
        const error = new Error(
          `OpenAI API error: ${response.status} ${response.statusText}`,
        );

        this.logApiError(
          durationMs,
          prompt_id,
          modelToUse,
          error,
          response.status,
        );
        throw error;
      }

      // For streaming, we'll log the response when the stream completes
      return this.parseStreamResponse(response, prompt_id, modelToUse);
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
    prompt_id: string,
    modelToUse: string,
  ): AsyncGenerator<GenerateContentResponse> {
    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('No response body');
    }

    const decoder = new TextDecoder();
    let buffer = '';

    // Accumulate tool calls across chunks
    const accumulatedToolCalls = new Map<
      string,
      {
        id: string;
        type: string;
        function: {
          name?: string;
          arguments: string;
        };
      }
    >();

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
              // Before finishing, yield any complete tool calls
              if (accumulatedToolCalls.size > 0) {
                const completedToolCalls = Array.from(
                  accumulatedToolCalls.values(),
                ).filter((tc) => tc.function.name); // Only require name, not arguments

                if (completedToolCalls.length > 0) {
                  const functionCalls: FunctionCall[] = [];
                  const functionCallParts: Part[] = [];

                  for (const toolCall of completedToolCalls) {
                    let args: Record<string, unknown> = {};
                    try {
                      if (toolCall.function.arguments) {
                        args = JSON.parse(toolCall.function.arguments);
                      }
                    } catch (_e) {
                      // If arguments aren't valid JSON, leave as empty object
                    }

                    const functionCall: FunctionCall = {
                      name: toolCall.function.name!,
                      args,
                      id: toolCall.id,
                    };

                    functionCalls.push(functionCall);
                    functionCallParts.push({ functionCall });
                  }

                  // Create response with function calls using the same pattern as non-streaming
                  const geminiResponse = {
                    candidates: [
                      {
                        index: 0,
                        content: {
                          role: 'model',
                          parts: functionCallParts,
                        },
                        finishReason: FinishReason.OTHER,
                        safetyRatings: [],
                      },
                    ],
                    modelVersion: modelToUse,
                    functionCalls: functionCalls,
                  };

                  yield geminiResponse as unknown as GenerateContentResponse;
                }
              }

              return;
            }

            try {
              const parsed: OpenAIStreamResponse = JSON.parse(data);

              // Debug logging for streaming chunks
              if (this.config?.getDebugMode()) {
                console.log(
                  `[DEBUG] OpenAI API Streaming Chunk (${prompt_id}):`,
                  JSON.stringify(parsed, null, 2),
                );
              }

              const choice = parsed.choices[0];
              if (choice?.delta) {
                const parts: Part[] = [];

                // Add text content if present
                if (choice.delta.content) {
                  parts.push({ text: choice.delta.content });
                }

                // Handle tool calls - accumulate across chunks
                if (
                  choice.delta.tool_calls &&
                  choice.delta.tool_calls.length > 0
                ) {
                  for (const deltaToolCall of choice.delta.tool_calls) {
                    // Use index as the primary key since it's consistent across chunks
                    const toolCallIndex = deltaToolCall.index ?? 0;
                    const toolCallId = `tool_${toolCallIndex}`;

                    // Get or create accumulated tool call
                    let accumulated = accumulatedToolCalls.get(toolCallId);
                    if (!accumulated) {
                      accumulated = {
                        id: deltaToolCall.id || toolCallId,
                        type: deltaToolCall.type || 'function',
                        function: {
                          name: undefined,
                          arguments: '',
                        },
                      };
                      accumulatedToolCalls.set(toolCallId, accumulated);
                    }

                    // Update tool call data
                    if (deltaToolCall.function) {
                      if (deltaToolCall.function.name) {
                        accumulated.function.name = deltaToolCall.function.name;
                      }
                      if (deltaToolCall.function.arguments) {
                        accumulated.function.arguments +=
                          deltaToolCall.function.arguments;
                      }
                    }

                    // Update the ID if we have a real ID from the first chunk
                    if (deltaToolCall.id) {
                      accumulated.id = deltaToolCall.id;
                    }
                  }
                }

                // Only yield if we have text content (tool calls are yielded at the end)
                if (parts.length > 0) {
                  const geminiResponse = new GenerateContentResponse();
                  geminiResponse.candidates = [
                    {
                      index: choice.index,
                      content: {
                        role: 'model',
                        parts,
                      },
                      finishReason:
                        choice.finish_reason === 'stop'
                          ? FinishReason.STOP
                          : choice.finish_reason
                            ? FinishReason.OTHER
                            : undefined,
                      safetyRatings: [],
                    },
                  ];
                  geminiResponse.modelVersion = parsed.model;

                  yield geminiResponse;
                }
              }
            } catch (_e) {
              // Skip invalid JSON
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }

  private trimThink(text: string): string {
    if (text) {
      const think = '</think>';
      const posThink = text.indexOf(think);
      if (posThink >= 0) {
        text = text.substring(posThink + think.length).trim();
      }
    }
    return text;
  }

  public preprocess(text?: string): string {
    // Default implementation just returns the text as is
    if (text) {
      text = this.trimThink(text);
      let posCode = text.indexOf('```json');
      if (posCode >= 0) {
        posCode += 7; // Skip past the ```json
        let posCodeEnd = text.indexOf('```', posCode);
        text = text
          .substring(posCode, posCodeEnd > 0 ? posCodeEnd : undefined)
          .trim();
      }
    }
    return text || '';
  }

  async countTokens(
    request: CountTokensParameters,
  ): Promise<CountTokensResponse> {
    // Extract system instruction from config if available
    const systemInstruction = this.extractTextFromContentUnion(
      request.config?.systemInstruction,
    );

    const messages = this.convertToOpenAIMessages(
      request.contents,
      systemInstruction,
    );

    // For OpenAI-compatible APIs, we estimate tokens based on character count
    // This is a rough approximation: ~4 characters per token for English text
    const totalText = messages.map((m) => m.content).join(' ');
    const estimatedTokens = Math.ceil(totalText.length / 4);

    return {
      totalTokens: estimatedTokens,
    };
  }

  async embedContent(
    request: EmbedContentParameters,
  ): Promise<EmbedContentResponse> {
    const startTime = Date.now();
    const prompt_id = 'openai-embed-content';

    // Extract text from contents
    const normalizedContents = this.normalizeContents(request.contents);
    const text = normalizedContents
      .flatMap((content) => content.parts || [])
      .filter(
        (part: Part) => part && typeof part === 'object' && 'text' in part,
      )
      .map((part: Part) => (part as { text: string }).text)
      .join('\n');

    const embeddingRequest = {
      model: request.model || this.model,
      input: text,
    };

    const requestUrl = `${this.baseUrl}/v1/embeddings`;

    // Log the request
    if (this.config?.getDebugMode()) {
      console.log(`[DEBUG] OpenAI Embedding API Request (${prompt_id}):`);
      console.log(`  URL: ${requestUrl}`);
      console.log(`  Headers:`, JSON.stringify(this.headers, null, 2));
      console.log(`  Request Body:`, JSON.stringify(embeddingRequest, null, 2));
    }

    try {
      const response = await fetch(requestUrl, {
        method: 'POST',
        headers: this.headers,
        body: JSON.stringify(embeddingRequest),
      });

      if (!response.ok) {
        const responseHeaders = this.extractResponseHeaders(response);

        // Log error with response details
        if (this.config?.getDebugMode()) {
          console.log(
            `[DEBUG] OpenAI Embedding API Error Response (${prompt_id}):`,
          );
          console.log(`  Status: ${response.status} ${response.statusText}`);
          console.log(
            `  Response Headers:`,
            JSON.stringify(responseHeaders, null, 2),
          );
        }

        throw new Error(
          `OpenAI API error: ${response.status} ${response.statusText}`,
        );
      }

      const responseHeaders = this.extractResponseHeaders(response);
      const embeddingResponse = await response.json();
      const embedding = embeddingResponse.data?.[0]?.embedding;

      // Log successful response
      const durationMs = Date.now() - startTime;
      if (this.config?.getDebugMode()) {
        console.log(`[DEBUG] OpenAI Embedding API Response (${prompt_id}):`);
        console.log(`  Duration: ${durationMs}ms`);
        console.log(`  Status Code: ${response.status}`);
        console.log(
          `  Response Headers:`,
          JSON.stringify(responseHeaders, null, 2),
        );
        console.log(
          `  Response Body:`,
          JSON.stringify(embeddingResponse, null, 2),
        );
      }

      if (!embedding) {
        throw new Error('No embedding returned from API');
      }

      return {
        embeddings: [
          {
            values: embedding,
          },
        ],
      };
    } catch (error) {
      const durationMs = Date.now() - startTime;
      if (this.config?.getDebugMode() && error instanceof Error) {
        console.log(`[DEBUG] OpenAI Embedding API Error (${prompt_id}):`);
        console.log(`  Duration: ${durationMs}ms`);
        console.log(`  Error: ${error.message}`);
      }
      throw error;
    }
  }
}
