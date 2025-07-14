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
import { ConversationManager } from './conversationManager.js';
import { ChatCompressionInfo } from './turn.js';

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

interface TokenLimits {
  maxTokens: number;
  maxOutputTokens: number;
  reserveTokens: number; // Tokens to reserve for output
}

interface MessagePriority {
  message: OpenAIMessage;
  priority: number; // Higher = more important
  estimatedTokens: number;
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

  // Conversation management for persistent state
  private conversationManager?: ConversationManager;

  // Token limits for different models
  private readonly modelTokenLimits = new Map<string, TokenLimits>([
    ['gpt-4', { maxTokens: 8192, maxOutputTokens: 4096, reserveTokens: 1000 }],
    [
      'gpt-4-32k',
      { maxTokens: 32768, maxOutputTokens: 4096, reserveTokens: 1000 },
    ],
    [
      'gpt-4-turbo',
      { maxTokens: 128000, maxOutputTokens: 4096, reserveTokens: 1000 },
    ],
    [
      'gpt-4o',
      { maxTokens: 128000, maxOutputTokens: 4096, reserveTokens: 1000 },
    ],
    [
      'gpt-3.5-turbo',
      { maxTokens: 16385, maxOutputTokens: 4096, reserveTokens: 1000 },
    ],
    [
      'claude-3-opus',
      { maxTokens: 200000, maxOutputTokens: 4096, reserveTokens: 1000 },
    ],
    [
      'claude-3-sonnet',
      { maxTokens: 200000, maxOutputTokens: 4096, reserveTokens: 1000 },
    ],
    [
      'claude-3-haiku',
      { maxTokens: 200000, maxOutputTokens: 4096, reserveTokens: 1000 },
    ],
    // Default fallback
    [
      'default',
      { maxTokens: 128000, maxOutputTokens: 4096, reserveTokens: 1000 },
    ],
  ]);

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

    // Initialize conversation manager if config is available
    if (this.config) {
      this.conversationManager = new ConversationManager(
        this.config,
        this, // Pass self as ContentGenerator
      );
    }
  }

  /**
   * Initialize or reset the conversation manager
   * @param initialHistory - Optional initial conversation history
   */
  initConversationManager(initialHistory?: Content[]): void {
    if (!this.config) {
      console.warn('Cannot initialize conversation manager without config');
      return;
    }

    this.conversationManager = new ConversationManager(
      this.config,
      this,
      {},
      initialHistory,
    );
  }

  /**
   * Get the conversation manager
   * @returns ConversationManager instance or undefined
   */
  getConversationManager(): ConversationManager | undefined {
    return this.conversationManager;
  }

  /**
   * Get conversation history with optional curation
   * @param curated - Whether to return curated history
   * @returns Conversation history or empty array if no manager
   */
  getConversationHistory(curated: boolean = false): Content[] {
    return this.conversationManager?.getHistory(curated) || [];
  }

  /**
   * Add content to conversation history
   * @param content - Content to add
   */
  addToConversationHistory(content: Content): void {
    this.conversationManager?.addHistory(content);
  }

  /**
   * Clear conversation history
   */
  clearConversationHistory(): void {
    this.conversationManager?.clearHistory();
  }

  /**
   * Try to compress conversation if needed
   * @param force - Force compression regardless of token count
   * @returns Compression info if compression was performed
   */
  async tryCompressConversation(
    force: boolean = false,
  ): Promise<ChatCompressionInfo | null> {
    if (!this.conversationManager) {
      return null;
    }

    return await this.conversationManager.tryCompress(this.model, { force });
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
   * Estimate token count for a text string
   * Uses a more accurate estimation based on OpenAI's tokenization patterns
   */
  private estimateTokens(text: string): number {
    if (!text) return 0;

    // More accurate token estimation:
    // - Average ~4 characters per token for English
    // - Add extra tokens for special characters, punctuation
    // - Account for encoding overhead
    const charCount = text.length;
    const wordCount = text.split(/\s+/).length;
    const specialChars = (text.match(/[^\w\s]/g) || []).length;

    // Base estimation: 4 chars per token, but adjust for:
    // - Words (each word boundary adds complexity)
    // - Special characters (often encoded as separate tokens)
    const baseTokens = Math.ceil(charCount / 4);
    const wordTokens = Math.ceil(wordCount * 0.1); // Word boundary overhead
    const specialTokens = Math.ceil(specialChars * 0.3); // Special char overhead

    return baseTokens + wordTokens + specialTokens;
  }

  /**
   * Estimate tokens for an OpenAI message
   */
  private estimateMessageTokens(message: OpenAIMessage): number {
    let tokens = 0;

    // Role overhead (each message has role metadata)
    tokens += 4;

    // Content tokens
    if (message.content) {
      tokens += this.estimateTokens(message.content);
    }

    // Tool calls overhead
    if (message.tool_calls && message.tool_calls.length > 0) {
      tokens += 10; // Base overhead for tool calls
      for (const toolCall of message.tool_calls) {
        if (toolCall.function.name) {
          tokens += this.estimateTokens(toolCall.function.name);
        }
        if (toolCall.function.arguments) {
          tokens += this.estimateTokens(toolCall.function.arguments);
        }
        tokens += 5; // Per tool call overhead
      }
    }

    return tokens;
  }

  /**
   * Get token limits for a specific model
   */
  private getTokenLimits(model: string): TokenLimits {
    // Try exact match first
    const limits = this.modelTokenLimits.get(model);
    if (limits) return limits;

    // Try partial matches for model variants
    for (const [modelPattern, modelLimits] of this.modelTokenLimits.entries()) {
      if (
        model.includes(modelPattern) ||
        modelPattern.includes(model.split('-')[0])
      ) {
        return modelLimits;
      }
    }

    // Fallback to default
    return this.modelTokenLimits.get('default')!;
  }

  /**
   * Prioritize messages based on importance
   */
  private prioritizeMessages(messages: OpenAIMessage[]): MessagePriority[] {
    return messages.map((message, index) => {
      let priority = 0;
      const estimatedTokens = this.estimateMessageTokens(message);

      // System messages are highest priority
      if (message.role === 'system') {
        priority = 1000;
      }
      // Recent messages are more important
      else if (index >= messages.length - 5) {
        priority = 500 + (index - (messages.length - 5)) * 50;
      }
      // Messages with tool calls are important
      else if (message.tool_calls && message.tool_calls.length > 0) {
        priority = 400;
      }
      // User messages are more important than assistant messages
      else if (message.role === 'user') {
        priority = 300;
      }
      // Assistant messages
      else {
        priority = 200;
      }

      // Reduce priority for very long messages (they're expensive)
      if (estimatedTokens > 1000) {
        priority -= Math.floor(estimatedTokens / 1000) * 50;
      }

      return {
        message,
        priority,
        estimatedTokens,
      };
    });
  }

  /**
   * Optimize messages to fit within token limits
   */
  private optimizeMessages(
    messages: OpenAIMessage[],
    maxTokens: number,
    tools?: OpenAITool[],
  ): OpenAIMessage[] {
    const limits = this.getTokenLimits(this.model);

    // Calculate tokens for tools
    let toolTokens = 0;
    if (tools && tools.length > 0) {
      toolTokens = tools.reduce((sum, tool) => {
        let tokens = 10; // Base tool overhead
        tokens += this.estimateTokens(tool.function.name);
        if (tool.function.description) {
          tokens += this.estimateTokens(tool.function.description);
        }
        if (tool.function.parameters) {
          tokens += this.estimateTokens(
            JSON.stringify(tool.function.parameters),
          );
        }
        return sum + tokens;
      }, 0);
    }

    // Available tokens for messages (reserve space for output and tools)
    const availableTokens =
      Math.min(maxTokens, limits.maxTokens) - limits.reserveTokens - toolTokens;

    if (availableTokens <= 0) {
      throw new Error(
        `Token limit too restrictive. Available: ${availableTokens}, Tools: ${toolTokens}, Reserve: ${limits.reserveTokens}`,
      );
    }

    // Prioritize messages
    const prioritized = this.prioritizeMessages(messages);

    // Always include system messages first
    const optimized: OpenAIMessage[] = [];
    let totalTokens = 0;

    // Add system messages (highest priority)
    for (const item of prioritized.filter((p) => p.message.role === 'system')) {
      if (totalTokens + item.estimatedTokens <= availableTokens) {
        optimized.push(item.message);
        totalTokens += item.estimatedTokens;
      }
    }

    // Add other messages by priority
    const nonSystemMessages = prioritized
      .filter((p) => p.message.role !== 'system')
      .sort((a, b) => b.priority - a.priority);

    for (const item of nonSystemMessages) {
      if (totalTokens + item.estimatedTokens <= availableTokens) {
        optimized.push(item.message);
        totalTokens += item.estimatedTokens;
      } else {
        // Try to truncate the message if it's too long
        const truncated = this.truncateMessage(
          item.message,
          availableTokens - totalTokens,
        );
        if (truncated && this.estimateMessageTokens(truncated) > 0) {
          optimized.push(truncated);
          totalTokens += this.estimateMessageTokens(truncated);
        }
      }
    }

    // Ensure messages are in chronological order (preserve conversation flow)
    const messageOrder = new Map(messages.map((msg, idx) => [msg, idx]));
    optimized.sort(
      (a, b) => (messageOrder.get(a) || 0) - (messageOrder.get(b) || 0),
    );

    if (this.config?.getDebugMode()) {
      console.log(
        `[DEBUG] Token optimization: ${messages.length} -> ${optimized.length} messages`,
      );
      console.log(
        `[DEBUG] Estimated tokens: ${totalTokens}/${availableTokens} (${toolTokens} for tools)`,
      );
    }

    return optimized;
  }

  /**
   * Truncate a message to fit within token limit
   */
  private truncateMessage(
    message: OpenAIMessage,
    maxTokens: number,
  ): OpenAIMessage | null {
    if (maxTokens <= 10) return null; // Not enough space for meaningful content

    if (!message.content) return message;

    // Reserve tokens for role and metadata
    const contentTokens = maxTokens - 10;
    if (contentTokens <= 0) return null;

    // Estimate how much content we can keep
    const currentTokens = this.estimateTokens(message.content);
    if (currentTokens <= contentTokens) return message;

    // Truncate content proportionally
    const ratio = contentTokens / currentTokens;
    const targetLength = Math.floor(message.content.length * ratio * 0.9); // 10% buffer

    if (targetLength <= 50) return null; // Too short to be useful

    // Try to truncate at word boundaries
    const words = message.content.split(' ');
    let truncated = '';
    let wordCount = 0;

    for (const word of words) {
      const testContent = truncated + (truncated ? ' ' : '') + word;
      if (this.estimateTokens(testContent) > contentTokens) break;
      truncated = testContent;
      wordCount++;
    }

    if (wordCount === 0) return null; // Couldn't fit even one word

    return {
      ...message,
      content: truncated + '...[truncated]',
    };
  }

  /**
   * Check if an error is due to token limits
   */
  private isTokenLimitError(error: Error, statusCode?: number): boolean {
    const message = error.message.toLowerCase();
    return (
      (statusCode === 400 &&
        message.includes('token') &&
        (message.includes('limit') ||
          message.includes('maximum') ||
          message.includes('exceeded') ||
          message.includes('too long'))) ||
      message.includes('context_length_exceeded')
    );
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

      // Extract text content
      const text = (content.parts || [])
        .filter(
          (part: Part) => part && typeof part === 'object' && 'text' in part,
        )
        .map((part: Part) => (part as { text: string }).text)
        .join('\n');

      // Extract function calls and convert to tool calls
      const functionCallParts = (content.parts || []).filter(
        (part: Part) =>
          part && typeof part === 'object' && 'functionCall' in part,
      );

      let tool_calls: OpenAIToolCall[] | undefined;
      if (functionCallParts.length > 0) {
        tool_calls = functionCallParts.map((part, index) => {
          const functionCall = (part as { functionCall: FunctionCall })
            .functionCall;
          return {
            id: functionCall.id || `call_${index}`,
            type: 'function' as const,
            function: {
              name: functionCall.name,
              arguments: JSON.stringify(functionCall.args || {}),
            },
          };
        });
      }

      // Create message - content can be null if there are only tool calls
      const messageContent = text.trim() || null;

      // Only add message if there's content or tool calls
      if (messageContent || tool_calls) {
        messages.push({
          role,
          content: messageContent,
          tool_calls,
        });
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
      parts.push(this.trimThink(choice.message.content));
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
      functionCalls,
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

    // Check if conversation compression is needed before processing
    if (this.conversationManager) {
      try {
        const compressionInfo =
          await this.conversationManager.tryCompress(modelToUse);
        if (compressionInfo && this.config?.getDebugMode()) {
          console.log(
            `[DEBUG] Conversation compressed from ${compressionInfo.originalTokenCount} to ${compressionInfo.newTokenCount} tokens`,
          );
        }
      } catch (compressionError) {
        if (this.config?.getDebugMode()) {
          console.log(
            `[DEBUG] Conversation compression failed: ${compressionError instanceof Error ? compressionError.message : 'Unknown error'}`,
          );
        }
      }
    }

    // Extract system instruction from config
    // const systemInstruction = this.extractTextFromContentUnion(request.config?.systemInstruction);
    let originalMessages: OpenAIMessage[];

    // Use conversation history if available, otherwise convert from request contents
    if (
      this.conversationManager &&
      this.conversationManager.getHistory().length > 0
    ) {
      // Get curated history and convert to OpenAI format
      const curatedHistory = this.conversationManager.getHistory(true);
      originalMessages = this.convertToOpenAIMessages(curatedHistory);

      // Add new user input from request if not already in history
      const requestMessages = this.convertToOpenAIMessages(request.contents);
      if (requestMessages.length > 0) {
        // Check if the last message in conversation is different from request
        const lastRequestMessage = requestMessages[requestMessages.length - 1];
        const lastHistoryMessage =
          originalMessages[originalMessages.length - 1];

        if (
          !lastHistoryMessage ||
          lastHistoryMessage.content !== lastRequestMessage.content ||
          lastHistoryMessage.role !== lastRequestMessage.role
        ) {
          originalMessages.push(...requestMessages);
        }
      }
    } else {
      originalMessages = this.convertToOpenAIMessages(request.contents);
    }

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

    // Optimize messages to fit within token limits
    const maxTokens = request.config?.maxOutputTokens || 4096;
    let messages: OpenAIMessage[];

    try {
      messages = this.optimizeMessages(originalMessages, maxTokens, tools);

      if (messages.length === 0) {
        throw new Error('All messages were filtered out due to token limits');
      }

      if (
        this.config?.getDebugMode() &&
        messages.length < originalMessages.length
      ) {
        console.log(
          `[DEBUG] Token optimization reduced messages from ${originalMessages.length} to ${messages.length}`,
        );
      }
    } catch (optimizationError) {
      // If optimization fails, try with original messages and let the API handle it
      if (this.config?.getDebugMode()) {
        console.log(
          `[DEBUG] Token optimization failed: ${optimizationError instanceof Error ? optimizationError.message : 'Unknown error'}, trying with original messages`,
        );
      }
      messages = originalMessages;
    }

    const openaiRequest = {
      model: modelToUse,
      messages,
      max_tokens: maxTokens,
      temperature: request.config?.temperature || 0.7,
      top_p: request.config?.topP || 0.9,
      stream: false,
      tools: tools && tools.length > 0 ? tools : undefined,
      tool_choice: tools && tools.length > 0 ? 'auto' : undefined,
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
        const errorText = await response
          .text()
          .catch(() => 'Failed to read error response');
        const error = new Error(
          `OpenAI API error: ${response.status} ${response.statusText} - ${errorText}`,
        );

        // Check if this is a token limit error and we haven't optimized yet
        if (
          this.isTokenLimitError(error, response.status) &&
          messages.length === originalMessages.length
        ) {
          if (this.config?.getDebugMode()) {
            console.log(
              `[DEBUG] Token limit error detected, attempting more aggressive optimization`,
            );
          }

          // Try more aggressive optimization
          try {
            const aggressivelyOptimized = this.optimizeMessages(
              originalMessages,
              Math.floor(maxTokens * 0.7), // Use 70% of max tokens
              tools,
            );

            if (aggressivelyOptimized.length > 0) {
              const retryRequest = {
                ...openaiRequest,
                messages: aggressivelyOptimized,
              };

              const retryResponse = await fetch(requestUrl, {
                method: 'POST',
                headers: this.headers,
                body: this.normalizeBody(JSON.stringify(retryRequest)),
                signal: request.config?.abortSignal,
              });

              if (retryResponse.ok) {
                const openaiResponse: OpenAIResponse =
                  await retryResponse.json();
                const geminiResponse =
                  this.convertToGeminiResponse(openaiResponse);

                // Record conversation turn if conversation manager is available
                if (this.conversationManager) {
                  try {
                    // Extract user input from request
                    const userInput = this.normalizeContents(request.contents);

                    // Extract model output from response
                    const modelOutput: Content[] = [];
                    if (
                      geminiResponse.candidates &&
                      geminiResponse.candidates.length > 0
                    ) {
                      const candidate = geminiResponse.candidates[0];
                      if (candidate.content) {
                        modelOutput.push(candidate.content);
                      }
                    }

                    // Record the conversation turn
                    if (userInput.length > 0) {
                      this.conversationManager.recordHistory(
                        userInput[userInput.length - 1], // Last user input
                        modelOutput,
                      );
                    }
                  } catch (recordError) {
                    if (this.config?.getDebugMode()) {
                      console.log(
                        `[DEBUG] Failed to record conversation: ${recordError instanceof Error ? recordError.message : 'Unknown error'}`,
                      );
                    }
                  }
                }

                return geminiResponse;
              }
            }
          } catch (retryError) {
            if (this.config?.getDebugMode()) {
              console.log(
                `[DEBUG] Aggressive optimization retry failed: ${retryError instanceof Error ? retryError.message : 'Unknown error'}`,
              );
            }
          }
        }

        // Log error with response details
        if (this.config?.getDebugMode()) {
          console.log(`[DEBUG] OpenAI API Error Response (${prompt_id}):`);
          console.log(`  Status: ${response.status} ${response.statusText}`);
          console.log(
            `  Response Headers:`,
            JSON.stringify(responseHeaders, null, 2),
          );
          console.log(`  Error Text: ${errorText}`);
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

      // Record conversation turn if conversation manager is available
      if (this.conversationManager) {
        try {
          // Extract user input from request
          const userInput = this.normalizeContents(request.contents);

          // Extract model output from response
          const modelOutput: Content[] = [];
          if (
            geminiResponse.candidates &&
            geminiResponse.candidates.length > 0
          ) {
            const candidate = geminiResponse.candidates[0];
            if (candidate.content) {
              modelOutput.push(candidate.content);
            }
          }

          // Record the conversation turn
          if (userInput.length > 0) {
            this.conversationManager.recordHistory(
              userInput[userInput.length - 1], // Last user input
              modelOutput,
            );
          }
        } catch (recordError) {
          if (this.config?.getDebugMode()) {
            console.log(
              `[DEBUG] Failed to record conversation: ${recordError instanceof Error ? recordError.message : 'Unknown error'}`,
            );
          }
        }
      }

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

    // Check if conversation compression is needed before processing
    if (this.conversationManager) {
      try {
        const compressionInfo =
          await this.conversationManager.tryCompress(modelToUse);
        if (compressionInfo && this.config?.getDebugMode()) {
          console.log(
            `[DEBUG] Conversation compressed from ${compressionInfo.originalTokenCount} to ${compressionInfo.newTokenCount} tokens`,
          );
        }
      } catch (compressionError) {
        if (this.config?.getDebugMode()) {
          console.log(
            `[DEBUG] Conversation compression failed: ${compressionError instanceof Error ? compressionError.message : 'Unknown error'}`,
          );
        }
      }
    }

    // Extract system instruction from config
    // const systemInstruction = this.extractTextFromContentUnion(request.config?.systemInstruction);
    let originalMessages: OpenAIMessage[];

    // Use conversation history if available, otherwise convert from request contents
    if (
      this.conversationManager &&
      this.conversationManager.getHistory().length > 0
    ) {
      // Get curated history and convert to OpenAI format
      const curatedHistory = this.conversationManager.getHistory(true);
      originalMessages = this.convertToOpenAIMessages(curatedHistory);

      // Add new user input from request if not already in history
      const requestMessages = this.convertToOpenAIMessages(request.contents);
      if (requestMessages.length > 0) {
        // Check if the last message in conversation is different from request
        const lastRequestMessage = requestMessages[requestMessages.length - 1];
        const lastHistoryMessage =
          originalMessages[originalMessages.length - 1];

        if (
          !lastHistoryMessage ||
          lastHistoryMessage.content !== lastRequestMessage.content ||
          lastHistoryMessage.role !== lastRequestMessage.role
        ) {
          originalMessages.push(...requestMessages);
        }
      }
    } else {
      originalMessages = this.convertToOpenAIMessages(request.contents);
    }

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

    // Optimize messages to fit within token limits
    const maxTokens = request.config?.maxOutputTokens || 4096;
    let messages: OpenAIMessage[];

    try {
      messages = this.optimizeMessages(originalMessages, maxTokens, tools);

      if (messages.length === 0) {
        throw new Error('All messages were filtered out due to token limits');
      }

      if (
        this.config?.getDebugMode() &&
        messages.length < originalMessages.length
      ) {
        console.log(
          `[DEBUG] Token optimization for streaming reduced messages from ${originalMessages.length} to ${messages.length}`,
        );
      }
    } catch (optimizationError) {
      // If optimization fails, try with original messages and let the API handle it
      if (this.config?.getDebugMode()) {
        console.log(
          `[DEBUG] Token optimization failed for streaming: ${optimizationError instanceof Error ? optimizationError.message : 'Unknown error'}, trying with original messages`,
        );
      }
      messages = originalMessages;
    }

    const openaiRequest = {
      model: modelToUse,
      messages,
      max_tokens: maxTokens,
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
        const errorText = await response
          .text()
          .catch(() => 'Failed to read error response');
        const error = new Error(
          `OpenAI API error: ${response.status} ${response.statusText} - ${errorText}`,
        );

        // Check if this is a token limit error and we haven't optimized yet
        if (
          this.isTokenLimitError(error, response.status) &&
          messages.length === originalMessages.length
        ) {
          if (this.config?.getDebugMode()) {
            console.log(
              `[DEBUG] Token limit error detected in streaming, attempting more aggressive optimization`,
            );
          }

          // Try more aggressive optimization
          try {
            const aggressivelyOptimized = this.optimizeMessages(
              originalMessages,
              Math.floor(maxTokens * 0.7), // Use 70% of max tokens
              tools,
            );

            if (aggressivelyOptimized.length > 0) {
              const retryRequest = {
                ...openaiRequest,
                messages: aggressivelyOptimized,
              };

              const retryResponse = await fetch(requestUrl, {
                method: 'POST',
                headers: this.headers,
                body: this.normalizeBody(JSON.stringify(retryRequest)),
                signal: request.config?.abortSignal,
              });

              if (retryResponse.ok) {
                return this.parseStreamResponse(
                  retryResponse,
                  prompt_id,
                  modelToUse,
                );
              }
            }
          } catch (retryError) {
            if (this.config?.getDebugMode()) {
              console.log(
                `[DEBUG] Aggressive optimization retry failed for streaming: ${retryError instanceof Error ? retryError.message : 'Unknown error'}`,
              );
            }
          }
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

      // For streaming, we'll log the response when the stream completes
      return this.wrapStreamResponseForConversationRecording(
        this.parseStreamResponse(response, prompt_id, modelToUse),
        request,
      );
    } catch (error) {
      const durationMs = Date.now() - startTime;
      if (error instanceof Error) {
        this.logApiError(durationMs, prompt_id, modelToUse, error);
      }
      throw error;
    }
  }

  /**
   * Wrap streaming response to record conversation when stream completes
   * @param streamGenerator - Original stream generator
   * @param request - Original request for extracting user input
   * @returns Wrapped stream generator that records conversation
   */
  private async *wrapStreamResponseForConversationRecording(
    streamGenerator: AsyncGenerator<GenerateContentResponse>,
    request: GenerateContentParameters,
  ): AsyncGenerator<GenerateContentResponse> {
    const modelOutputParts: Part[] = [];
    let lastResponse: GenerateContentResponse | undefined;

    for await (const chunk of streamGenerator) {
      lastResponse = chunk;

      // Collect content parts from the stream
      if (chunk.candidates && chunk.candidates.length > 0) {
        const candidate = chunk.candidates[0];
        if (candidate.content && candidate.content.parts) {
          modelOutputParts.push(...candidate.content.parts);
        }
      }

      yield chunk;
    }

    // Record conversation after stream completes
    if (this.conversationManager && lastResponse) {
      try {
        // Extract user input from request
        const userInput = this.normalizeContents(request.contents);

        // Create model output content from collected parts
        const modelOutput: Content[] = [];
        if (modelOutputParts.length > 0) {
          modelOutput.push({
            role: 'model',
            parts: modelOutputParts,
          });
        }

        // Record the conversation turn
        if (userInput.length > 0) {
          this.conversationManager.recordHistory(
            userInput[userInput.length - 1], // Last user input
            modelOutput,
          );
        }
      } catch (recordError) {
        if (this.config?.getDebugMode()) {
          console.log(
            `[DEBUG] Failed to record streaming conversation: ${recordError instanceof Error ? recordError.message : 'Unknown error'}`,
          );
        }
      }
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
                    functionCalls,
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

  private trimThink(text: string): { text: string; thought: boolean } {
    let thought = false;
    if (text) {
      const think = '</think>';
      const posThink = text.indexOf(think);
      if (posThink >= 0) {
        thought = true;
        text = text.substring(posThink + think.length).trim();
      }
    }
    return { text, thought };
  }

  preprocess(text?: string): string {
    // Default implementation just returns the text as is
    if (text) {
      const tt = this.trimThink(text);
      let posCode = tt.text.indexOf('```json');
      if (posCode >= 0) {
        posCode += 7; // Skip past the ```json
        const posCodeEnd = tt.text.indexOf('```', posCode);
        text = tt.text
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

    // Use our improved token estimation
    const totalTokens = messages.reduce(
      (sum, message) => sum + this.estimateMessageTokens(message),
      0,
    );

    // Add tokens for tools if present
    let toolTokens = 0;
    if (request.config?.tools && Array.isArray(request.config.tools)) {
      const geminiTools = request.config.tools.filter(
        (tool): tool is Tool =>
          tool && typeof tool === 'object' && 'functionDeclarations' in tool,
      );
      const openaiTools = this.convertGeminiToolsToOpenAI(geminiTools);

      if (openaiTools && openaiTools.length > 0) {
        toolTokens = openaiTools.reduce((sum, tool) => {
          let tokens = 10; // Base tool overhead
          tokens += this.estimateTokens(tool.function.name);
          if (tool.function.description) {
            tokens += this.estimateTokens(tool.function.description);
          }
          if (tool.function.parameters) {
            tokens += this.estimateTokens(
              JSON.stringify(tool.function.parameters),
            );
          }
          return sum + tokens;
        }, 0);
      }
    }

    const finalTokenCount = totalTokens + toolTokens;

    if (this.config?.getDebugMode()) {
      console.log(
        `[DEBUG] Token count estimation: ${totalTokens} message tokens + ${toolTokens} tool tokens = ${finalTokenCount} total`,
      );
    }

    return {
      totalTokens: finalTokenCount,
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
