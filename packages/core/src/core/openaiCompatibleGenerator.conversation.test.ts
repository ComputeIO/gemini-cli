/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { OpenAICompatibleGenerator } from './openaiCompatibleGenerator.js';
import { Config } from '../config/config.js';

// Mock fetch globally
const mockFetch = vi.fn();
global.fetch = mockFetch;

// Mock the tokenLimit function
vi.mock('./tokenLimits.js', () => ({
  tokenLimit: vi.fn(() => 100000),
}));

describe('OpenAICompatibleGenerator - Conversation Management', () => {
  let generator: OpenAICompatibleGenerator;
  let config: Config;

  beforeEach(() => {
    vi.clearAllMocks();

    // Mock config with debug mode
    config = {
      getDebugMode: () => false,
      getTelemetryEnabled: () => false,
      getTelemetryLogPromptsEnabled: () => false,
      getUsageStatisticsEnabled: () => false,
      getSessionId: () => 'test-session',
      getModel: () => 'test-model',
    } as Config;

    generator = new OpenAICompatibleGenerator(
      'http://localhost:11434',
      'test-model',
      undefined,
      {},
      config,
    );
  });

  describe('conversation manager initialization', () => {
    it('should initialize conversation manager when config is provided', () => {
      const conversationManager = generator.getConversationManager();
      expect(conversationManager).toBeDefined();
    });

    it('should not initialize conversation manager without config', () => {
      const generatorWithoutConfig = new OpenAICompatibleGenerator(
        'http://localhost:11434',
        'test-model',
      );

      const conversationManager =
        generatorWithoutConfig.getConversationManager();
      expect(conversationManager).toBeUndefined();
    });

    it('should allow manual initialization of conversation manager', () => {
      generator.initConversationManager();
      const conversationManager = generator.getConversationManager();
      expect(conversationManager).toBeDefined();
    });
  });

  describe('conversation history management', () => {
    it('should add content to conversation history', () => {
      const content = {
        role: 'user' as const,
        parts: [{ text: 'Hello' }],
      };

      generator.addToConversationHistory(content);
      const history = generator.getConversationHistory();

      expect(history).toHaveLength(1);
      expect(history[0]).toEqual(content);
    });

    it('should get conversation history with curation option', () => {
      const content = {
        role: 'user' as const,
        parts: [{ text: 'Hello' }],
      };

      generator.addToConversationHistory(content);

      const regularHistory = generator.getConversationHistory(false);
      const curatedHistory = generator.getConversationHistory(true);

      expect(regularHistory).toEqual(curatedHistory);
      expect(regularHistory).toHaveLength(1);
    });

    it('should clear conversation history', () => {
      const content = {
        role: 'user' as const,
        parts: [{ text: 'Hello' }],
      };

      generator.addToConversationHistory(content);
      generator.clearConversationHistory();

      const history = generator.getConversationHistory();
      expect(history).toHaveLength(0);
    });
  });

  describe('conversation compression', () => {
    it('should try to compress conversation', async () => {
      // Mock the conversation manager's compression method
      const conversationManager = generator.getConversationManager();
      if (conversationManager) {
        const mockTryCompress = vi
          .spyOn(conversationManager, 'tryCompress')
          .mockResolvedValue({
            originalTokenCount: 80000,
            newTokenCount: 30000,
          });

        const result = await generator.tryCompressConversation();

        expect(mockTryCompress).toHaveBeenCalledWith('test-model', {
          force: false,
        });
        expect(result).toEqual({
          originalTokenCount: 80000,
          newTokenCount: 30000,
        });
      } else {
        throw new Error('Conversation manager not initialized');
      }
    });

    it('should force compression when requested', async () => {
      const conversationManager = generator.getConversationManager();
      if (conversationManager) {
        const mockTryCompress = vi
          .spyOn(conversationManager, 'tryCompress')
          .mockResolvedValue(null);

        const result = await generator.tryCompressConversation(true);

        expect(mockTryCompress).toHaveBeenCalledWith('test-model', {
          force: true,
        });
        expect(result).toBeNull();
      } else {
        throw new Error('Conversation manager not initialized');
      }
    });

    it('should return null when no conversation manager', async () => {
      const generatorWithoutConfig = new OpenAICompatibleGenerator(
        'http://localhost:11434',
        'test-model',
      );

      const result = await generatorWithoutConfig.tryCompressConversation();
      expect(result).toBeNull();
    });
  });

  describe('generateContent with conversation management', () => {
    it('should record conversation turn after successful generation', async () => {
      const mockResponse = {
        id: 'test-id',
        object: 'chat.completion',
        created: Date.now(),
        model: 'test-model',
        choices: [
          {
            index: 0,
            message: {
              role: 'assistant',
              content: 'Hello, how can I help you?',
            },
            finish_reason: 'stop',
          },
        ],
        usage: {
          prompt_tokens: 10,
          completion_tokens: 7,
          total_tokens: 17,
        },
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const request = {
        model: 'test-model',
        contents: [
          {
            role: 'user' as const,
            parts: [{ text: 'Hello' }],
          },
        ],
      };

      const conversationManager = generator.getConversationManager();
      if (conversationManager) {
        const mockRecordHistory = vi.spyOn(
          conversationManager,
          'recordHistory',
        );

        await generator.generateContent(request);

        expect(mockRecordHistory).toHaveBeenCalled();

        const recordCall = mockRecordHistory.mock.calls[0];
        expect(recordCall[0]).toEqual(request.contents[0]); // User input
        expect(recordCall[1]).toHaveLength(1); // Model output array
        expect(recordCall[1][0].role).toBe('model');
      } else {
        throw new Error('Conversation manager not initialized');
      }
    });

    it('should check for compression before generation', async () => {
      const mockResponse = {
        id: 'test-id',
        object: 'chat.completion',
        created: Date.now(),
        model: 'test-model',
        choices: [
          {
            index: 0,
            message: {
              role: 'assistant',
              content: 'Hello!',
            },
            finish_reason: 'stop',
          },
        ],
        usage: {
          prompt_tokens: 5,
          completion_tokens: 2,
          total_tokens: 7,
        },
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const request = {
        model: 'test-model',
        contents: [
          {
            role: 'user' as const,
            parts: [{ text: 'Hi' }],
          },
        ],
      };

      const conversationManager = generator.getConversationManager();
      if (conversationManager) {
        const mockTryCompress = vi
          .spyOn(conversationManager, 'tryCompress')
          .mockResolvedValue(null);

        await generator.generateContent(request);

        expect(mockTryCompress).toHaveBeenCalledWith('test-model');
      } else {
        throw new Error('Conversation manager not initialized');
      }
    });

    it('should use conversation history when available', async () => {
      // Add some history first
      generator.addToConversationHistory({
        role: 'user',
        parts: [{ text: 'Previous message' }],
      });
      generator.addToConversationHistory({
        role: 'model',
        parts: [{ text: 'Previous response' }],
      });

      const mockResponse = {
        id: 'test-id',
        object: 'chat.completion',
        created: Date.now(),
        model: 'test-model',
        choices: [
          {
            index: 0,
            message: {
              role: 'assistant',
              content: 'Current response',
            },
            finish_reason: 'stop',
          },
        ],
        usage: {
          prompt_tokens: 20,
          completion_tokens: 5,
          total_tokens: 25,
        },
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const request = {
        model: 'test-model',
        contents: [
          {
            role: 'user' as const,
            parts: [{ text: 'Current message' }],
          },
        ],
      };

      await generator.generateContent(request);

      // Verify the request included the conversation history
      const requestBody = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(requestBody.messages).toHaveLength(3); // Previous user + model + current user
      expect(requestBody.messages[0].content).toBe('Previous message');
      expect(requestBody.messages[1].content).toBe('Previous response');
      expect(requestBody.messages[2].content).toBe('Current message');
    });
  });

  describe('generateContentStream with conversation management', () => {
    it('should record conversation after streaming completes', async () => {
      const mockStreamData = [
        'data: {"choices":[{"delta":{"content":"Hello"}}]}',
        'data: {"choices":[{"delta":{"content":" there"}}]}',
        'data: [DONE]',
      ];

      const mockResponse = {
        ok: true,
        body: {
          getReader: () => ({
            read: vi
              .fn()
              .mockResolvedValueOnce({
                done: false,
                value: new TextEncoder().encode(mockStreamData[0] + '\n'),
              })
              .mockResolvedValueOnce({
                done: false,
                value: new TextEncoder().encode(mockStreamData[1] + '\n'),
              })
              .mockResolvedValueOnce({
                done: false,
                value: new TextEncoder().encode(mockStreamData[2] + '\n'),
              })
              .mockResolvedValueOnce({
                done: true,
                value: undefined,
              }),
            releaseLock: vi.fn(),
          }),
        },
      };

      mockFetch.mockResolvedValueOnce(mockResponse);

      const request = {
        model: 'test-model',
        contents: [
          {
            role: 'user' as const,
            parts: [{ text: 'Hello' }],
          },
        ],
      };

      const conversationManager = generator.getConversationManager();
      if (conversationManager) {
        const mockRecordHistory = vi.spyOn(
          conversationManager,
          'recordHistory',
        );

        const streamGenerator = await generator.generateContentStream(request);

        // Consume the entire stream
        const chunks = [];
        for await (const chunk of streamGenerator) {
          chunks.push(chunk);
        }

        expect(mockRecordHistory).toHaveBeenCalled();

        const recordCall = mockRecordHistory.mock.calls[0];
        expect(recordCall[0]).toEqual(request.contents[0]); // User input
        expect(recordCall[1]).toHaveLength(1); // Model output array
        expect(recordCall[1][0].role).toBe('model');
      } else {
        throw new Error('Conversation manager not initialized');
      }
    });
  });
});
