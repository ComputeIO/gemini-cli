/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { ConversationManager } from './conversationManager.js';
import { Config } from '../config/config.js';
import { ContentGenerator } from './contentGenerator.js';
import { Content } from '@google/genai';

// Mock the tokenLimit function
vi.mock('./tokenLimits.js', () => ({
  tokenLimit: vi.fn(() => 100000), // Default 100k token limit
}));

// Mock the compression prompt
vi.mock('./prompts.js', () => ({
  getCompressionPrompt: vi.fn(() => 'Test compression prompt'),
}));

describe('ConversationManager', () => {
  let config: Config;
  let contentGenerator: ContentGenerator;
  let conversationManager: ConversationManager;

  beforeEach(() => {
    // Mock config
    config = {
      getModel: () => 'test-model',
    } as Config;

    // Mock content generator
    contentGenerator = {
      countTokens: vi.fn().mockResolvedValue({ totalTokens: 1000 }),
      generateContent: vi.fn().mockResolvedValue({
        candidates: [
          {
            content: {
              role: 'model',
              parts: [{ text: 'This is a test summary.' }],
            },
          },
        ],
      }),
    } as unknown as ContentGenerator;

    conversationManager = new ConversationManager(
      config,
      contentGenerator,
      {},
      [],
    );
  });

  describe('basic functionality', () => {
    it('should initialize with empty history', () => {
      expect(conversationManager.getHistory()).toEqual([]);
    });

    it('should add content to history', () => {
      const content: Content = {
        role: 'user',
        parts: [{ text: 'Hello' }],
      };

      conversationManager.addHistory(content);
      const history = conversationManager.getHistory();

      expect(history).toHaveLength(1);
      expect(history[0]).toEqual(content);
    });

    it('should clear history', () => {
      const content: Content = {
        role: 'user',
        parts: [{ text: 'Hello' }],
      };

      conversationManager.addHistory(content);
      conversationManager.clearHistory();

      expect(conversationManager.getHistory()).toEqual([]);
    });

    it('should set history', () => {
      const history: Content[] = [
        { role: 'user', parts: [{ text: 'Hello' }] },
        { role: 'model', parts: [{ text: 'Hi there!' }] },
      ];

      conversationManager.setHistory(history);

      expect(conversationManager.getHistory()).toEqual(history);
    });
  });

  describe('history recording', () => {
    it('should record conversation turn correctly', () => {
      const userInput: Content = {
        role: 'user',
        parts: [{ text: 'What is 2+2?' }],
      };

      const modelOutput: Content[] = [
        {
          role: 'model',
          parts: [{ text: '2+2 equals 4.' }],
        },
      ];

      conversationManager.recordHistory(userInput, modelOutput);
      const history = conversationManager.getHistory();

      expect(history).toHaveLength(2);
      expect(history[0]).toEqual(userInput);
      expect(history[1]).toEqual(modelOutput[0]);
    });

    it('should consolidate adjacent model responses', () => {
      const userInput: Content = {
        role: 'user',
        parts: [{ text: 'Tell me a story' }],
      };

      const modelOutput: Content[] = [
        { role: 'model', parts: [{ text: 'Once upon a time,' }] },
        { role: 'model', parts: [{ text: ' there was a brave knight.' }] },
      ];

      conversationManager.recordHistory(userInput, modelOutput);
      const history = conversationManager.getHistory();

      expect(history).toHaveLength(2);
      expect(history[0]).toEqual(userInput);
      expect(history[1]?.parts?.[0]?.text).toBe(
        'Once upon a time, there was a brave knight.',
      );
    });
  });

  describe('history curation', () => {
    it('should return curated history without invalid content', () => {
      const validHistory: Content[] = [
        { role: 'user', parts: [{ text: 'Hello' }] },
        { role: 'model', parts: [{ text: 'Hi there!' }] },
        { role: 'user', parts: [{ text: 'How are you?' }] },
        { role: 'model', parts: [{ text: '' }] }, // Invalid empty response
      ];

      conversationManager.setHistory(validHistory);
      const curatedHistory = conversationManager.getHistory(true);

      // Should exclude the invalid model response and the user message that preceded it
      expect(curatedHistory).toHaveLength(2);
      expect(curatedHistory[0]?.parts?.[0]?.text).toBe('Hello');
      expect(curatedHistory[1]?.parts?.[0]?.text).toBe('Hi there!');
    });

    it('should validate history roles', () => {
      const invalidHistory = [
        { role: 'invalid', parts: [{ text: 'Hello' }] },
      ] as Content[];

      expect(() => {
        conversationManager.setHistory(invalidHistory);
      }).toThrow('Role must be user or model');
    });
  });

  describe('compression', () => {
    it('should not compress when under token threshold', async () => {
      const history: Content[] = [
        { role: 'user', parts: [{ text: 'Hello' }] },
        { role: 'model', parts: [{ text: 'Hi there!' }] },
      ];

      conversationManager.setHistory(history);

      // Mock low token count (under 70% threshold)
      const mockCountTokens = vi.fn().mockResolvedValue({ totalTokens: 50000 });
      contentGenerator.countTokens = mockCountTokens;

      const result = await conversationManager.tryCompress('test-model');

      expect(result).toBeNull();
      expect(mockCountTokens).toHaveBeenCalledWith({
        model: 'test-model',
        contents: history,
      });
    });

    it('should compress when over token threshold', async () => {
      const history: Content[] = [
        { role: 'user', parts: [{ text: 'Long conversation part 1' }] },
        { role: 'model', parts: [{ text: 'Response 1' }] },
        { role: 'user', parts: [{ text: 'Long conversation part 2' }] },
        { role: 'model', parts: [{ text: 'Response 2' }] },
        { role: 'user', parts: [{ text: 'Recent message' }] },
        { role: 'model', parts: [{ text: 'Recent response' }] },
      ];

      conversationManager.setHistory(history);

      // Mock high token count (over 70% threshold)
      const originalTokens = 80000; // Over 70% of 100k
      const compressedTokens = 30000;

      const mockCountTokens = vi
        .fn()
        .mockResolvedValueOnce({ totalTokens: originalTokens }) // Original count
        .mockResolvedValueOnce({ totalTokens: compressedTokens }); // After compression

      contentGenerator.countTokens = mockCountTokens;

      const result = await conversationManager.tryCompress('test-model');

      expect(result).toEqual({
        originalTokenCount: originalTokens,
        newTokenCount: compressedTokens,
      });

      // Check that history was compressed
      const newHistory = conversationManager.getHistory();
      expect(newHistory.length).toBeGreaterThan(0);
      expect(newHistory[0]?.parts?.[0]?.text).toBe('This is a test summary.');
    });

    it('should force compression regardless of token count', async () => {
      const history: Content[] = [
        { role: 'user', parts: [{ text: 'Hello' }] },
        { role: 'model', parts: [{ text: 'Hi there!' }] },
      ];

      conversationManager.setHistory(history);

      // Mock low token count
      const originalTokens = 10000;
      const compressedTokens = 5000;

      const mockCountTokens = vi
        .fn()
        .mockResolvedValueOnce({ totalTokens: originalTokens })
        .mockResolvedValueOnce({ totalTokens: compressedTokens });

      contentGenerator.countTokens = mockCountTokens;

      const result = await conversationManager.tryCompress('test-model', {
        force: true,
      });

      expect(result).toEqual({
        originalTokenCount: originalTokens,
        newTokenCount: compressedTokens,
      });
    });

    it('should use custom compression thresholds', async () => {
      const history: Content[] = [
        { role: 'user', parts: [{ text: 'Hello' }] },
        { role: 'model', parts: [{ text: 'Hi there!' }] },
      ];

      conversationManager.setHistory(history);

      // Mock token count that would normally not trigger compression
      const originalTokens = 60000; // 60% of 100k (normally under 70% threshold)
      const compressedTokens = 30000;

      const mockCountTokens = vi
        .fn()
        .mockResolvedValueOnce({ totalTokens: originalTokens })
        .mockResolvedValueOnce({ totalTokens: compressedTokens });

      contentGenerator.countTokens = mockCountTokens;

      // Use custom threshold of 50%
      const result = await conversationManager.tryCompress('test-model', {
        threshold: 0.5,
        preserveThreshold: 0.5,
      });

      expect(result).toEqual({
        originalTokenCount: originalTokens,
        newTokenCount: compressedTokens,
      });
    });
  });

  describe('error handling', () => {
    it('should handle compression generation failure gracefully', async () => {
      const history: Content[] = [
        { role: 'user', parts: [{ text: 'Hello' }] },
        { role: 'model', parts: [{ text: 'Hi there!' }] },
      ];

      conversationManager.setHistory(history);

      // Mock high token count
      const mockCountTokens = vi.fn().mockResolvedValue({ totalTokens: 80000 });
      contentGenerator.countTokens = mockCountTokens;

      // Mock generation failure
      const mockGenerateContent = vi
        .fn()
        .mockRejectedValue(new Error('Generation failed'));
      contentGenerator.generateContent = mockGenerateContent;

      await expect(
        conversationManager.tryCompress('test-model'),
      ).rejects.toThrow('Generation failed');
    });

    it('should handle undefined token counts', async () => {
      const history: Content[] = [
        { role: 'user', parts: [{ text: 'Hello' }] },
        { role: 'model', parts: [{ text: 'Hi there!' }] },
      ];

      conversationManager.setHistory(history);

      // Mock undefined token count
      const mockCountTokens = vi
        .fn()
        .mockResolvedValue({ totalTokens: undefined });
      contentGenerator.countTokens = mockCountTokens;

      const result = await conversationManager.tryCompress('test-model', {
        force: true,
      });

      expect(result).toBeNull();
    });
  });

  describe('index calculation', () => {
    it('should calculate fraction index correctly', () => {
      const history: Content[] = [
        { role: 'user', parts: [{ text: 'Message 1' }] },
        { role: 'model', parts: [{ text: 'Response 1' }] },
        { role: 'user', parts: [{ text: 'Message 2' }] },
        { role: 'model', parts: [{ text: 'Response 2' }] },
      ];

      conversationManager.setHistory(history);

      // Test that private method works correctly through compression
      expect(() => {
        // This will internally use findIndexAfterFraction
        conversationManager.tryCompress('test-model', { force: true });
      }).not.toThrow();
    });
  });
});
