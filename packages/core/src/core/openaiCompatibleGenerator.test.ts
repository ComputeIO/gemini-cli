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
});
