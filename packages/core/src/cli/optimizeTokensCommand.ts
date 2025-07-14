/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Config } from '../config/config.js';
import { OpenAICompatibleGenerator } from '../core/openaiCompatibleGenerator.js';
import { ConversationManager } from '../core/conversationManager.js';

interface OptimizeTokensOptions {
  /** Force compression regardless of token count */
  force?: boolean;
  /** Show detailed compression information */
  verbose?: boolean;
  /** Clear conversation history */
  clear?: boolean;
  /** Show current conversation statistics */
  stats?: boolean;
  /** Custom compression threshold (0-1) */
  threshold?: number;
  /** Custom preservation threshold (0-1) */
  preserve?: number;
}

/**
 * Command to optimize token usage in conversations by managing conversation history,
 * compressing old content, and providing statistics about token usage.
 */
export function optimizeTokensForGenerator(
  generator: OpenAICompatibleGenerator,
  config: Config,
  options: OptimizeTokensOptions = {},
): Promise<void> {
  return optimizeTokens(generator, config, options);
}

async function optimizeTokens(
  generator: OpenAICompatibleGenerator,
  config: Config,
  options: OptimizeTokensOptions = {},
): Promise<void> {
  // Check if the generator supports conversation management
  let conversationManager: ConversationManager | undefined;

  conversationManager = generator.getConversationManager();
  if (!conversationManager) {
    generator.initConversationManager();
    conversationManager = generator.getConversationManager();
  }

  if (!conversationManager) {
    console.log(
      'Conversation management not available for this generator type.',
    );
    return;
  }

  // Handle clear option
  if (options.clear) {
    conversationManager.clearHistory();
    console.log('✅ Conversation history cleared.');
    return;
  }

  // Show statistics
  if (options.stats || options.verbose) {
    await showConversationStats(
      conversationManager,
      config.getModel(),
      options.verbose,
    );
  }

  // Perform compression if requested or needed
  if (
    options.force ||
    options.threshold !== undefined ||
    options.preserve !== undefined
  ) {
    const compressionOptions = {
      force: options.force,
      threshold: options.threshold,
      preserveThreshold: options.preserve,
    };

    console.log('🔄 Attempting conversation compression...');

    const compressionResult = await conversationManager.tryCompress(
      config.getModel(),
      compressionOptions,
    );

    if (compressionResult) {
      const { originalTokenCount, newTokenCount } = compressionResult;
      const savedTokens = originalTokenCount - newTokenCount;
      const percentageSaved = (
        (savedTokens / originalTokenCount) *
        100
      ).toFixed(1);

      console.log('✅ Compression successful!');
      console.log(`   Original tokens: ${originalTokenCount.toLocaleString()}`);
      console.log(`   New tokens: ${newTokenCount.toLocaleString()}`);
      console.log(
        `   Saved: ${savedTokens.toLocaleString()} tokens (${percentageSaved}%)`,
      );

      if (options.verbose) {
        await showConversationStats(
          conversationManager,
          config.getModel(),
          true,
        );
      }
    } else {
      if (options.force) {
        console.log('ℹ️  No compression needed or compression failed.');
      } else {
        console.log('ℹ️  Conversation is within acceptable token limits.');
        console.log('   Use --force to compress anyway.');
      }
    }
  }

  // If no specific action was requested, show help
  if (
    !options.force &&
    !options.clear &&
    !options.stats &&
    !options.verbose &&
    options.threshold === undefined &&
    options.preserve === undefined
  ) {
    console.log('Token Optimization Commands:');
    console.log('  --stats       Show conversation statistics');
    console.log('  --force       Force compression');
    console.log('  --clear       Clear conversation history');
    console.log('  --threshold   Set compression threshold (0-1)');
    console.log('  --preserve    Set preservation threshold (0-1)');
    console.log('  --verbose     Show detailed information');
    console.log('');
    console.log('Example usage:');
    console.log('  gemini optimize-tokens --stats');
    console.log('  gemini optimize-tokens --force --verbose');
    console.log('  gemini optimize-tokens --threshold 0.6 --preserve 0.4');
  }
}

async function showConversationStats(
  conversationManager: ConversationManager,
  model: string,
  verbose: boolean = false,
): Promise<void> {
  const history = conversationManager.getHistory();
  const curatedHistory = conversationManager.getHistory(true);

  console.log('📊 Conversation Statistics:');
  console.log(`   Total messages: ${history.length}`);
  console.log(`   Curated messages: ${curatedHistory.length}`);

  if (history.length > 0) {
    try {
      // Count tokens for current history
      const tokenResponse = await conversationManager.countTokens(
        model,
        history,
      );
      const currentTokens = tokenResponse.totalTokens || 0;

      // Get model token limit
      const tokenLimitForModel = getModelTokenLimit(model);
      const usagePercentage = (
        (currentTokens / tokenLimitForModel) *
        100
      ).toFixed(1);

      console.log(`   Current tokens: ${currentTokens.toLocaleString()}`);
      console.log(`   Token limit: ${tokenLimitForModel.toLocaleString()}`);
      console.log(`   Usage: ${usagePercentage}%`);

      // Determine status
      if (currentTokens > tokenLimitForModel * 0.9) {
        console.log('   Status: 🔴 Critical - Compression highly recommended');
      } else if (currentTokens > tokenLimitForModel * 0.7) {
        console.log('   Status: 🟡 Warning - Consider compression');
      } else {
        console.log('   Status: 🟢 Good - Within normal limits');
      }

      if (verbose && curatedHistory.length !== history.length) {
        const curatedTokenResponse = await conversationManager.countTokens(
          model,
          curatedHistory,
        );
        const curatedTokens = curatedTokenResponse.totalTokens || 0;
        const savedTokens = currentTokens - curatedTokens;

        console.log('');
        console.log('📝 Curation Impact:');
        console.log(`   Curated tokens: ${curatedTokens.toLocaleString()}`);
        console.log(
          `   Tokens saved by curation: ${savedTokens.toLocaleString()}`,
        );
      }
    } catch (error) {
      console.log('   Token count: Unable to calculate');
      if (verbose) {
        console.log(`   Error: ${error}`);
      }
    }
  }

  if (verbose && history.length > 0) {
    console.log('');
    console.log('📋 Message Breakdown:');

    const messageCounts = history.reduce(
      (counts, content) => {
        const role = content.role;
        if (role) {
          counts[role] = (counts[role] || 0) + 1;
        }
        return counts;
      },
      {} as Record<string, number>,
    );

    Object.entries(messageCounts).forEach(([role, count]) => {
      console.log(`   ${role}: ${count} messages`);
    });
  }
}

function getModelTokenLimit(model: string): number {
  // Token limits for different models (should match those in OpenAICompatibleGenerator)
  const limits: Record<string, number> = {
    'gpt-4': 8192,
    'gpt-4-32k': 32768,
    'gpt-4-turbo': 128000,
    'gpt-4o': 128000,
    'gpt-3.5-turbo': 16385,
    'claude-3-opus': 200000,
    'claude-3-sonnet': 200000,
    'claude-3-haiku': 200000,
  };

  return limits[model] || 128000; // Default fallback
}
