/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { OpenAICompatibleGenerator } from './openaiCompatibleGenerator.js';

// Mock fetch globally
global.fetch = vi.fn();

describe('OpenAICompatibleGenerator', () => {
  let generator: OpenAICompatibleGenerator;
  const mockFetch = global.fetch as ReturnType<typeof vi.fn>;

  beforeEach(() => {
    generator = new OpenAICompatibleGenerator(
      'http://localhost:11434',
      'deepseek-r1:latest',
      undefined,
      {}
    );
    mockFetch.mockClear();
  });

  describe('generateContent', () => {
    it('should generate content successfully', async () => {
      const mockResponse = {
        id: 'test-id',
        object: 'chat.completion',
        created: Date.now(),
        model: 'deepseek-r1:latest',
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
        model: 'deepseek-r1:latest',
        contents: [
          {
            role: 'user' as const,
            parts: [{ text: 'Hello' }],
          },
        ],
      };

      const result = await generator.generateContent(request);

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:11434/v1/chat/completions',
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            'Content-Type': 'application/json',
          }),
          body: expect.stringContaining('"messages"'),
        })
      );

      expect(result.candidates).toHaveLength(1);
      expect(result.candidates![0].content!.parts![0]).toEqual({
        text: 'Hello, how can I help you?',
      });
      expect(result.usageMetadata!.totalTokenCount).toBe(17);
    });

    it('should handle API errors', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
        statusText: 'Not Found',
      });

      const request = {
        model: 'deepseek-r1:latest',
        contents: [
          {
            role: 'user' as const,
            parts: [{ text: 'Hello' }],
          },
        ],
      };

      await expect(generator.generateContent(request)).rejects.toThrow(
        'OpenAI API error: 404 Not Found'
      );
    });
  });

  describe('countTokens', () => {
    it('should estimate token count', async () => {
      const request = {
        model: 'deepseek-r1:latest',
        contents: [
          {
            role: 'user' as const,
            parts: [{ text: 'Hello world' }],
          },
        ],
      };

      const result = await generator.countTokens(request);

      expect(result.totalTokens).toBeGreaterThan(0);
      expect(typeof result.totalTokens).toBe('number');
    });
  });

  describe('embedContent', () => {
    it('should generate embeddings', async () => {
      const mockEmbeddingResponse = {
        data: [
          {
            embedding: [0.1, 0.2, 0.3, 0.4],
            index: 0,
            object: 'embedding',
          },
        ],
        model: 'text-embedding-ada-002',
        object: 'list',
        usage: {
          prompt_tokens: 8,
          total_tokens: 8,
        },
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockEmbeddingResponse),
      });

      const request = {
        model: 'text-embedding-ada-002',
        contents: [
          {
            role: 'user' as const,
            parts: [{ text: 'Hello world' }],
          },
        ],
      };

      const result = await generator.embedContent(request);

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:11434/v1/embeddings',
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            'Content-Type': 'application/json',
          }),
          body: expect.stringContaining('"input"'),
        })
      );

      expect(result.embeddings).toHaveLength(1);
      expect(result.embeddings![0].values).toEqual([0.1, 0.2, 0.3, 0.4]);
    });

    it('should handle missing embedding in response', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ data: [] }),
      });

      const request = {
        model: 'text-embedding-ada-002',
        contents: [
          {
            role: 'user' as const,
            parts: [{ text: 'Hello world' }],
          },
        ],
      };

      await expect(generator.embedContent(request)).rejects.toThrow(
        'No embedding returned from API'
      );
    });
  });

  describe('debug logging', () => {
    it('should log request and response bodies when debug mode is enabled', async () => {
      // Mock console.log to capture debug output
      const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});

      // Mock config with debug mode enabled
      const mockConfig = {
        getDebugMode: () => true,
        getTelemetryEnabled: () => false,
        getTelemetryLogPromptsEnabled: () => false,
        getUsageStatisticsEnabled: () => true,
        getSessionId: () => 'test-session',
      };

      const generatorWithDebug = new OpenAICompatibleGenerator(
        'http://localhost:11434',
        'deepseek-r1:latest',
        undefined,
        {},
        mockConfig as any
      );

      const mockResponse = {
        id: 'test-id',
        object: 'chat.completion',
        created: Date.now(),
        model: 'deepseek-r1:latest',
        choices: [
          {
            index: 0,
            message: {
              role: 'assistant',
              content: 'Debug test response',
            },
            finish_reason: 'stop',
          },
        ],
        usage: {
          prompt_tokens: 5,
          completion_tokens: 3,
          total_tokens: 8,
        },
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const request = {
        model: 'deepseek-r1:latest',
        contents: [
          {
            role: 'user' as const,
            parts: [{ text: 'Debug test' }],
          },
        ],
      };

      await generatorWithDebug.generateContent(request);

      // Check that debug logs were called with URL, headers, and request/response details
      expect(consoleSpy).toHaveBeenCalledWith('[DEBUG] OpenAI API Request (openai-generate-content):');
      expect(consoleSpy).toHaveBeenCalledWith('  URL: http://localhost:11434/v1/chat/completions');
      expect(consoleSpy).toHaveBeenCalledWith('  Headers:', expect.stringContaining('"Content-Type": "application/json"'));
      expect(consoleSpy).toHaveBeenCalledWith('  Request Body:', expect.stringContaining('"model": "deepseek-r1:latest"'));
      
      expect(consoleSpy).toHaveBeenCalledWith('[DEBUG] OpenAI API Response (openai-generate-content):');
      expect(consoleSpy).toHaveBeenCalledWith(expect.stringMatching(/  Duration: \d+ms/));
      expect(consoleSpy).toHaveBeenCalledWith('  Status Code: 200');
      expect(consoleSpy).toHaveBeenCalledWith('  Response Headers:', expect.any(String));
      expect(consoleSpy).toHaveBeenCalledWith('  Response Body:', expect.stringContaining('"content": "Debug test response"'));

      consoleSpy.mockRestore();
    });

    it('should not log debug info when debug mode is disabled', async () => {
      // Mock console.log to capture debug output
      const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});

      // Mock config with debug mode disabled
      const mockConfig = {
        getDebugMode: () => false,
        getTelemetryEnabled: () => false,
        getTelemetryLogPromptsEnabled: () => false,
        getUsageStatisticsEnabled: () => true,
        getSessionId: () => 'test-session',
      };

      const generatorWithoutDebug = new OpenAICompatibleGenerator(
        'http://localhost:11434',
        'deepseek-r1:latest',
        undefined,
        {},
        mockConfig as any
      );

      const mockResponse = {
        id: 'test-id',
        object: 'chat.completion',
        created: Date.now(),
        model: 'deepseek-r1:latest',
        choices: [
          {
            index: 0,
            message: {
              role: 'assistant',
              content: 'No debug response',
            },
            finish_reason: 'stop',
          },
        ],
        usage: {
          prompt_tokens: 5,
          completion_tokens: 3,
          total_tokens: 8,
        },
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const request = {
        model: 'deepseek-r1:latest',
        contents: [
          {
            role: 'user' as const,
            parts: [{ text: 'No debug test' }],
          },
        ],
      };

      await generatorWithoutDebug.generateContent(request);

      // Check that no debug logs were called
      expect(consoleSpy).not.toHaveBeenCalledWith(
        '[DEBUG] OpenAI API Request (openai-generate-content):',
        expect.anything()
      );
      expect(consoleSpy).not.toHaveBeenCalledWith(
        '[DEBUG] OpenAI API Response (openai-generate-content):',
        expect.anything()
      );

      consoleSpy.mockRestore();
    });
  });
});
