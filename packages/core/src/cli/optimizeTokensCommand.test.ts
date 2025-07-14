/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { optimizeTokensForGenerator } from './optimizeTokensCommand.js';
import { OpenAICompatibleGenerator } from '../core/openaiCompatibleGenerator.js';
import { ConversationManager } from '../core/conversationManager.js';
import { Config } from '../config/config.js';

// Mock console methods
const mockConsoleLog = vi.spyOn(console, 'log').mockImplementation(() => {});

describe('optimizeTokensCommand', () => {
  let generator: OpenAICompatibleGenerator;
  let config: Config;
  let conversationManager: ConversationManager;

  beforeEach(() => {
    vi.clearAllMocks();

    // Mock config
    config = {
      getModel: () => 'test-model',
    } as Config;

    // Mock conversation manager
    conversationManager = {
      clearHistory: vi.fn(),
      getHistory: vi.fn().mockReturnValue([]),
      tryCompress: vi.fn(),
      countTokens: vi.fn().mockResolvedValue({ totalTokens: 1000 }),
    } as unknown as ConversationManager;

    // Mock generator
    generator = {
      getConversationManager: vi.fn().mockReturnValue(conversationManager),
      initConversationManager: vi.fn(),
    } as unknown as OpenAICompatibleGenerator;
  });

  describe('basic functionality', () => {
    it('should initialize conversation manager if not available', async () => {
      const generatorWithoutManager = {
        getConversationManager: vi.fn().mockReturnValue(undefined),
        initConversationManager: vi.fn(),
      } as unknown as OpenAICompatibleGenerator;

      generatorWithoutManager.getConversationManager = vi
        .fn()
        .mockReturnValueOnce(undefined)
        .mockReturnValueOnce(conversationManager);

      await optimizeTokensForGenerator(generatorWithoutManager, config, {
        stats: true,
      });

      expect(
        generatorWithoutManager.initConversationManager,
      ).toHaveBeenCalled();
    });

    it('should handle generator without conversation management', async () => {
      const generatorWithoutManager = {
        getConversationManager: vi.fn().mockReturnValue(undefined),
        initConversationManager: vi.fn(),
      } as unknown as OpenAICompatibleGenerator;

      await optimizeTokensForGenerator(generatorWithoutManager, config, {
        stats: true,
      });

      expect(mockConsoleLog).toHaveBeenCalledWith(
        'Conversation management not available for this generator type.',
      );
    });
  });

  describe('clear functionality', () => {
    it('should clear conversation history when clear option is provided', async () => {
      await optimizeTokensForGenerator(generator, config, { clear: true });

      expect(conversationManager.clearHistory).toHaveBeenCalled();
      expect(mockConsoleLog).toHaveBeenCalledWith(
        '✅ Conversation history cleared.',
      );
    });
  });

  describe('stats functionality', () => {
    it('should show conversation statistics when stats option is provided', async () => {
      const mockHistory = [
        { role: 'user', parts: [{ text: 'Hello' }] },
        { role: 'model', parts: [{ text: 'Hi there!' }] },
      ];

      conversationManager.getHistory = vi
        .fn()
        .mockReturnValueOnce(mockHistory)
        .mockReturnValueOnce(mockHistory); // Called twice for curated history

      await optimizeTokensForGenerator(generator, config, { stats: true });

      expect(mockConsoleLog).toHaveBeenCalledWith(
        '📊 Conversation Statistics:',
      );
      expect(mockConsoleLog).toHaveBeenCalledWith('   Total messages: 2');
      expect(mockConsoleLog).toHaveBeenCalledWith('   Curated messages: 2');
    });

    it('should show token usage statistics', async () => {
      const mockHistory = [{ role: 'user', parts: [{ text: 'Hello' }] }];

      conversationManager.getHistory = vi.fn().mockReturnValue(mockHistory);
      conversationManager.countTokens = vi
        .fn()
        .mockResolvedValue({ totalTokens: 50000 });

      await optimizeTokensForGenerator(generator, config, { stats: true });

      expect(mockConsoleLog).toHaveBeenCalledWith('   Current tokens: 50,000');
      expect(mockConsoleLog).toHaveBeenCalledWith('   Token limit: 128,000');
      expect(mockConsoleLog).toHaveBeenCalledWith('   Usage: 39.1%');
      expect(mockConsoleLog).toHaveBeenCalledWith(
        '   Status: 🟢 Good - Within normal limits',
      );
    });

    it('should show warning status for high token usage', async () => {
      const mockHistory = [{ role: 'user', parts: [{ text: 'Hello' }] }];

      conversationManager.getHistory = vi.fn().mockReturnValue(mockHistory);
      conversationManager.countTokens = vi
        .fn()
        .mockResolvedValue({ totalTokens: 90000 });

      await optimizeTokensForGenerator(generator, config, { stats: true });

      expect(mockConsoleLog).toHaveBeenCalledWith(
        '   Status: 🟡 Warning - Consider compression',
      );
    });

    it('should show critical status for very high token usage', async () => {
      const mockHistory = [{ role: 'user', parts: [{ text: 'Hello' }] }];

      conversationManager.getHistory = vi.fn().mockReturnValue(mockHistory);
      conversationManager.countTokens = vi
        .fn()
        .mockResolvedValue({ totalTokens: 120000 });

      await optimizeTokensForGenerator(generator, config, { stats: true });

      expect(mockConsoleLog).toHaveBeenCalledWith(
        '   Status: 🔴 Critical - Compression highly recommended',
      );
    });

    it('should handle token counting errors gracefully', async () => {
      const mockHistory = [{ role: 'user', parts: [{ text: 'Hello' }] }];

      conversationManager.getHistory = vi.fn().mockReturnValue(mockHistory);
      conversationManager.countTokens = vi
        .fn()
        .mockRejectedValue(new Error('Token counting failed'));

      await optimizeTokensForGenerator(generator, config, {
        stats: true,
        verbose: true,
      });

      expect(mockConsoleLog).toHaveBeenCalledWith(
        '   Token count: Unable to calculate',
      );
      expect(mockConsoleLog).toHaveBeenCalledWith(
        '   Error: Error: Token counting failed',
      );
    });
  });

  describe('compression functionality', () => {
    it('should perform compression when force option is provided', async () => {
      const compressionResult = {
        originalTokenCount: 80000,
        newTokenCount: 30000,
      };

      conversationManager.tryCompress = vi
        .fn()
        .mockResolvedValue(compressionResult);

      await optimizeTokensForGenerator(generator, config, { force: true });

      expect(conversationManager.tryCompress).toHaveBeenCalledWith(
        'test-model',
        {
          force: true,
          threshold: undefined,
          preserveThreshold: undefined,
        },
      );

      expect(mockConsoleLog).toHaveBeenCalledWith(
        '🔄 Attempting conversation compression...',
      );
      expect(mockConsoleLog).toHaveBeenCalledWith('✅ Compression successful!');
      expect(mockConsoleLog).toHaveBeenCalledWith('   Original tokens: 80,000');
      expect(mockConsoleLog).toHaveBeenCalledWith('   New tokens: 30,000');
      expect(mockConsoleLog).toHaveBeenCalledWith(
        '   Saved: 50,000 tokens (62.5%)',
      );
    });

    it('should use custom compression thresholds', async () => {
      conversationManager.tryCompress = vi.fn().mockResolvedValue(null);

      await optimizeTokensForGenerator(generator, config, {
        threshold: 0.6,
        preserve: 0.4,
      });

      expect(conversationManager.tryCompress).toHaveBeenCalledWith(
        'test-model',
        {
          force: undefined,
          threshold: 0.6,
          preserveThreshold: 0.4,
        },
      );
    });

    it('should handle failed compression', async () => {
      conversationManager.tryCompress = vi.fn().mockResolvedValue(null);

      await optimizeTokensForGenerator(generator, config, { force: true });

      expect(mockConsoleLog).toHaveBeenCalledWith(
        'ℹ️  No compression needed or compression failed.',
      );
    });

    it('should handle compression not needed', async () => {
      conversationManager.tryCompress = vi.fn().mockResolvedValue(null);

      await optimizeTokensForGenerator(generator, config, { threshold: 0.8 });

      expect(mockConsoleLog).toHaveBeenCalledWith(
        'ℹ️  Conversation is within acceptable token limits.',
      );
      expect(mockConsoleLog).toHaveBeenCalledWith(
        '   Use --force to compress anyway.',
      );
    });
  });

  describe('verbose mode', () => {
    it('should show detailed information in verbose mode', async () => {
      const mockHistory = [
        { role: 'user', parts: [{ text: 'Hello' }] },
        { role: 'model', parts: [{ text: 'Hi' }] },
        { role: 'user', parts: [{ text: 'How are you?' }] },
      ];

      conversationManager.getHistory = vi.fn().mockReturnValue(mockHistory);
      conversationManager.countTokens = vi
        .fn()
        .mockResolvedValue({ totalTokens: 1000 });

      await optimizeTokensForGenerator(generator, config, { verbose: true });

      expect(mockConsoleLog).toHaveBeenCalledWith('📋 Message Breakdown:');
      expect(mockConsoleLog).toHaveBeenCalledWith('   user: 2 messages');
      expect(mockConsoleLog).toHaveBeenCalledWith('   model: 1 messages');
    });

    it('should show curation impact in verbose mode', async () => {
      const fullHistory = [
        { role: 'user', parts: [{ text: 'Hello' }] },
        { role: 'model', parts: [{ text: 'Hi' }] },
      ];
      const curatedHistory = [{ role: 'user', parts: [{ text: 'Hello' }] }];

      conversationManager.getHistory = vi
        .fn()
        .mockReturnValueOnce(fullHistory)
        .mockReturnValueOnce(curatedHistory)
        .mockReturnValue(curatedHistory);

      conversationManager.countTokens = vi
        .fn()
        .mockResolvedValueOnce({ totalTokens: 1000 })
        .mockResolvedValueOnce({ totalTokens: 500 });

      await optimizeTokensForGenerator(generator, config, { verbose: true });

      expect(mockConsoleLog).toHaveBeenCalledWith('📝 Curation Impact:');
      expect(mockConsoleLog).toHaveBeenCalledWith('   Curated tokens: 500');
      expect(mockConsoleLog).toHaveBeenCalledWith(
        '   Tokens saved by curation: 500',
      );
    });
  });

  describe('help display', () => {
    it('should show help when no options are provided', async () => {
      await optimizeTokensForGenerator(generator, config, {});

      expect(mockConsoleLog).toHaveBeenCalledWith(
        'Token Optimization Commands:',
      );
      expect(mockConsoleLog).toHaveBeenCalledWith(
        '  --stats       Show conversation statistics',
      );
      expect(mockConsoleLog).toHaveBeenCalledWith(
        '  --force       Force compression',
      );
      expect(mockConsoleLog).toHaveBeenCalledWith(
        '  --clear       Clear conversation history',
      );
      expect(mockConsoleLog).toHaveBeenCalledWith('Example usage:');
    });
  });

  describe('conversation management validation', () => {
    it('should maintain conversation history like default content generator', () => {
      // Create a real generator to test actual conversation manager behavior
      const realConfig = {
        getModel: () => 'test-model',
      } as Config;

      const realGenerator = new OpenAICompatibleGenerator(
        'http://localhost:11434',
        'test-model',
        undefined,
        {},
        realConfig,
      );

      const realConversationManager = realGenerator.getConversationManager();
      expect(realConversationManager).toBeDefined();

      // Add user input
      const userInput = {
        role: 'user' as const,
        parts: [{ text: 'Hello' }],
      };

      // Add model output
      const modelOutput = [
        {
          role: 'model' as const,
          parts: [{ text: 'Hi there!' }],
        },
      ];

      // Record history like GeminiChat does
      realConversationManager!.recordHistory(userInput, modelOutput);

      const history = realConversationManager!.getHistory();
      expect(history).toHaveLength(2);
      expect(history[0]).toEqual(userInput);
      expect(history[1]).toEqual(modelOutput[0]);
    });

    it('should handle curated history like GeminiChat', () => {
      const realConfig = {
        getModel: () => 'test-model',
      } as Config;

      const realGenerator = new OpenAICompatibleGenerator(
        'http://localhost:11434',
        'test-model',
        undefined,
        {},
        realConfig,
      );

      const realConversationManager = realGenerator.getConversationManager();

      // Add valid content
      realConversationManager!.addHistory({
        role: 'user',
        parts: [{ text: 'Valid user message' }],
      });

      realConversationManager!.addHistory({
        role: 'model',
        parts: [{ text: 'Valid model response' }],
      });

      // Add invalid content (empty text)
      realConversationManager!.addHistory({
        role: 'user',
        parts: [{ text: 'Another user message' }],
      });

      realConversationManager!.addHistory({
        role: 'model',
        parts: [{ text: '' }], // Invalid: empty text
      });

      const fullHistory = realConversationManager!.getHistory(false);
      const curatedHistory = realConversationManager!.getHistory(true);

      expect(fullHistory).toHaveLength(4);
      expect(curatedHistory).toHaveLength(2); // Invalid model response and preceding user message removed
    });

    it('should consolidate adjacent model responses like GeminiChat', () => {
      const realConfig = {
        getModel: () => 'test-model',
      } as Config;

      const realGenerator = new OpenAICompatibleGenerator(
        'http://localhost:11434',
        'test-model',
        undefined,
        {},
        realConfig,
      );

      const realConversationManager = realGenerator.getConversationManager();

      const userInput = {
        role: 'user' as const,
        parts: [{ text: 'Tell me a story' }],
      };

      const modelOutputs = [
        {
          role: 'model' as const,
          parts: [{ text: 'Once upon a time, ' }],
        },
        {
          role: 'model' as const,
          parts: [{ text: 'there was a brave knight.' }],
        },
      ];

      realConversationManager!.recordHistory(userInput, modelOutputs);

      const history = realConversationManager!.getHistory();
      expect(history).toHaveLength(2); // User input + consolidated model response
      expect(history[1].parts).toBeDefined();
      expect(history[1].parts?.length).toBeGreaterThan(0);
      const modelPart = history[1].parts?.[0];
      expect(modelPart).toBeDefined();
      expect('text' in modelPart!).toBe(true);
      expect((modelPart as { text: string }).text).toBe(
        'Once upon a time, there was a brave knight.',
      );
    });

    it('should handle thought content like GeminiChat', () => {
      const realConfig = {
        getModel: () => 'test-model',
      } as Config;

      const realGenerator = new OpenAICompatibleGenerator(
        'http://localhost:11434',
        'test-model',
        undefined,
        {},
        realConfig,
      );

      const realConversationManager = realGenerator.getConversationManager();

      const userInput = {
        role: 'user' as const,
        parts: [{ text: 'What do you think?' }],
      };

      const modelOutputs = [
        {
          role: 'model' as const,
          parts: [{ thought: true }], // Thought content should be filtered
        },
        {
          role: 'model' as const,
          parts: [{ text: 'I think this is interesting.' }],
        },
      ];

      realConversationManager!.recordHistory(userInput, modelOutputs);

      const history = realConversationManager!.getHistory();
      expect(history).toHaveLength(2); // User input + non-thought model response
      const modelPart = history[1].parts?.[0];
      expect(modelPart).toBeDefined();
      expect('text' in modelPart!).toBe(true);
      expect((modelPart as { text: string }).text).toBe(
        'I think this is interesting.',
      );
    });
  });
});
