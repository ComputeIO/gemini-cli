/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  Content,
  Part,
  GenerateContentConfig,
  CountTokensResponse,
} from '@google/genai';
import { ContentGenerator } from './contentGenerator.js';
import { Config } from '../config/config.js';
import { tokenLimit } from './tokenLimits.js';
import { getCompressionPrompt } from './prompts.js';
import { isFunctionResponse } from '../utils/messageInspectors.js';
import { ChatCompressionInfo } from './turn.js';

/**
 * Options for conversation compression
 */
export interface CompressionOptions {
  /** Force compression regardless of token count */
  force?: boolean;
  /** Custom compression threshold (0-1) */
  threshold?: number;
  /** Custom preservation threshold (0-1) */
  preserveThreshold?: number;
}

/**
 * Conversation history manager that provides compression and curation capabilities
 * for maintaining long conversations within token limits.
 *
 * Based on GeminiChat's proven compression strategies adapted for OpenAI-compatible generators.
 */
export class ConversationManager {
  private history: Content[] = [];

  /**
   * Default threshold for compression token count as a fraction of the model's token limit.
   * If the chat history exceeds this threshold, it will be compressed.
   */
  private readonly COMPRESSION_TOKEN_THRESHOLD = 0.7;

  /**
   * The fraction of the latest chat history to keep. A value of 0.3
   * means that only the last 30% of the chat history will be kept after compression.
   */
  private readonly COMPRESSION_PRESERVE_THRESHOLD = 0.3;

  constructor(
    private readonly config: Config,
    private readonly contentGenerator: ContentGenerator,
    private readonly generationConfig: GenerateContentConfig = {},
    initialHistory: Content[] = [],
  ) {
    this.validateHistory(initialHistory);
    this.history = structuredClone(initialHistory);
  }

  /**
   * Get the conversation history
   * @param curated - Whether to return curated (valid) history or comprehensive history
   * @returns Deep copy of the conversation history
   */
  getHistory(curated: boolean = false): Content[] {
    const history = curated
      ? this.extractCuratedHistory(this.history)
      : this.history;
    return structuredClone(history);
  }

  /**
   * Set the conversation history
   * @param history - New conversation history
   */
  setHistory(history: Content[]): void {
    this.validateHistory(history);
    this.history = structuredClone(history);
  }

  /**
   * Add content to the conversation history
   * @param content - Content to add
   */
  addHistory(content: Content): void {
    this.history.push(structuredClone(content));
  }

  /**
   * Clear the conversation history
   */
  clearHistory(): void {
    this.history = [];
  }

  /**
   * Record a conversation turn with proper consolidation
   * @param userInput - User's input content
   * @param modelOutput - Model's response content(s)
   * @param automaticFunctionCallingHistory - Additional function calling history
   */
  recordHistory(
    userInput: Content,
    modelOutput: Content[],
    automaticFunctionCallingHistory?: Content[],
  ): void {
    const nonThoughtModelOutput = modelOutput.filter(
      (content) => !this.isThoughtContent(content),
    );

    let outputContents: Content[] = [];
    if (
      nonThoughtModelOutput.length > 0 &&
      nonThoughtModelOutput.every((content) => content.role !== undefined)
    ) {
      outputContents = nonThoughtModelOutput;
    } else if (nonThoughtModelOutput.length === 0 && modelOutput.length > 0) {
      // This case handles when the model returns only a thought.
      // We don't want to add an empty model response in this case.
    } else {
      // When not a function response appends an empty content when model returns empty response, so that the
      // history is always alternating between user and model.
      if (!isFunctionResponse(userInput)) {
        outputContents.push({
          role: 'model',
          parts: [],
        } as Content);
      }
    }

    if (
      automaticFunctionCallingHistory &&
      automaticFunctionCallingHistory.length > 0
    ) {
      this.history.push(
        ...this.extractCuratedHistory(automaticFunctionCallingHistory),
      );
    } else {
      this.history.push(userInput);
    }

    // Consolidate adjacent model roles in outputContents
    const consolidatedOutputContents: Content[] = [];
    for (const content of outputContents) {
      if (this.isThoughtContent(content)) {
        continue;
      }
      const lastContent =
        consolidatedOutputContents[consolidatedOutputContents.length - 1];
      if (this.isTextContent(lastContent) && this.isTextContent(content)) {
        // If both current and last are text, combine their text into the lastContent's first part
        // and append any other parts from the current content.
        lastContent.parts[0].text += content.parts[0].text || '';
        if (content.parts.length > 1) {
          lastContent.parts.push(...content.parts.slice(1));
        }
      } else {
        consolidatedOutputContents.push(content);
      }
    }

    if (consolidatedOutputContents.length > 0) {
      const lastHistoryEntry = this.history[this.history.length - 1];
      const canMergeWithLastHistory =
        !automaticFunctionCallingHistory ||
        automaticFunctionCallingHistory.length === 0;

      if (
        canMergeWithLastHistory &&
        this.isTextContent(lastHistoryEntry) &&
        this.isTextContent(consolidatedOutputContents[0])
      ) {
        // If both current and last are text, combine their text into the lastHistoryEntry's first part
        // and append any other parts from the current content.
        lastHistoryEntry.parts[0].text +=
          consolidatedOutputContents[0].parts[0].text || '';
        if (consolidatedOutputContents[0].parts.length > 1) {
          lastHistoryEntry.parts.push(
            ...consolidatedOutputContents[0].parts.slice(1),
          );
        }
        consolidatedOutputContents.shift(); // Remove the first element as it's merged
      }
      this.history.push(...consolidatedOutputContents);
    }
  }

  /**
   * Check if compression is needed and optionally perform it
   * @param model - Model name to use for token counting
   * @param options - Compression options
   * @returns Compression info if compression was performed, null otherwise
   */
  async tryCompress(
    model: string,
    options: CompressionOptions = {},
  ): Promise<ChatCompressionInfo | null> {
    const {
      force = false,
      threshold = this.COMPRESSION_TOKEN_THRESHOLD,
      preserveThreshold = this.COMPRESSION_PRESERVE_THRESHOLD,
    } = options;

    const curatedHistory = this.extractCuratedHistory(this.history);

    // Don't do anything if the history is empty
    if (curatedHistory.length === 0) {
      return null;
    }

    const { totalTokens: originalTokenCount } =
      await this.contentGenerator.countTokens({
        model,
        contents: curatedHistory,
      });

    if (originalTokenCount === undefined) {
      console.warn(`Could not determine token count for model ${model}.`);
      return null;
    }

    // Don't compress if not forced and we are under the limit
    if (!force && originalTokenCount < threshold * tokenLimit(model)) {
      return null;
    }

    let compressBeforeIndex = this.findIndexAfterFraction(
      curatedHistory,
      1 - preserveThreshold,
    );

    // Find the first user message after the index. This is the start of the next turn.
    while (
      compressBeforeIndex < curatedHistory.length &&
      (curatedHistory[compressBeforeIndex]?.role === 'model' ||
        isFunctionResponse(curatedHistory[compressBeforeIndex]))
    ) {
      compressBeforeIndex++;
    }

    const historyToCompress = curatedHistory.slice(0, compressBeforeIndex);
    const historyToKeep = curatedHistory.slice(compressBeforeIndex);

    // Create temporary history for compression
    const tempHistory = structuredClone(historyToCompress);

    // Generate compression summary
    const summary = await this.generateCompressionSummary(tempHistory, model);

    // Create new history with compression
    this.history = [
      {
        role: 'user',
        parts: [{ text: summary }],
      },
      {
        role: 'model',
        parts: [{ text: 'Got it. Thanks for the additional context!' }],
      },
      ...historyToKeep,
    ];

    const { totalTokens: newTokenCount } =
      await this.contentGenerator.countTokens({
        model,
        contents: this.getHistory(),
      });

    if (newTokenCount === undefined) {
      console.warn('Could not determine compressed history token count.');
      return null;
    }

    return {
      originalTokenCount,
      newTokenCount,
    };
  }

  /**
   * Count tokens for a given model and content
   * @param model - Model name
   * @param contents - Content to count tokens for
   * @returns Token count response
   */
  async countTokens(
    model: string,
    contents: Content[],
  ): Promise<CountTokensResponse> {
    return await this.contentGenerator.countTokens({
      model,
      contents,
    });
  }

  /**
   * Generate a compression summary for the given history
   * @param historyToCompress - History to compress
   * @param model - Model to use for generating the summary
   * @returns Compression summary text
   */
  private async generateCompressionSummary(
    historyToCompress: Content[],
    model: string,
  ): Promise<string> {
    // Create a temporary conversation manager for compression
    const tempManager = new ConversationManager(
      this.config,
      this.contentGenerator,
      this.generationConfig,
      historyToCompress,
    );

    // Generate content with compression prompt
    const response = await this.contentGenerator.generateContent({
      model,
      contents: tempManager.getHistory(),
      config: {
        ...this.generationConfig,
        systemInstruction: { text: getCompressionPrompt() },
      },
    });

    const candidate = response.candidates?.[0];
    if (!candidate?.content?.parts) {
      throw new Error('Failed to generate compression summary');
    }

    // Extract text from the response
    const text = candidate.content.parts
      .filter((part) => part && typeof part === 'object' && 'text' in part)
      .map((part) => (part as { text: string }).text)
      .join('\n');

    return text.trim();
  }

  /**
   * Validates the history contains the correct roles
   * @param history - History to validate
   */
  private validateHistory(history: Content[]): void {
    for (const content of history) {
      if (content.role !== 'user' && content.role !== 'model') {
        throw new Error(`Role must be user or model, but got ${content.role}.`);
      }
    }
  }

  /**
   * Extracts the curated (valid) history from a comprehensive history
   * @param comprehensiveHistory - Full history including invalid entries
   * @returns Curated history with only valid entries
   */
  private extractCuratedHistory(comprehensiveHistory: Content[]): Content[] {
    if (
      comprehensiveHistory === undefined ||
      comprehensiveHistory.length === 0
    ) {
      return [];
    }

    const curatedHistory: Content[] = [];
    const length = comprehensiveHistory.length;
    let i = 0;

    while (i < length) {
      if (comprehensiveHistory[i].role === 'user') {
        curatedHistory.push(comprehensiveHistory[i]);
        i++;
      } else {
        const modelOutput: Content[] = [];
        let isValid = true;
        while (i < length && comprehensiveHistory[i].role === 'model') {
          modelOutput.push(comprehensiveHistory[i]);
          if (isValid && !this.isValidContent(comprehensiveHistory[i])) {
            isValid = false;
          }
          i++;
        }
        if (isValid) {
          curatedHistory.push(...modelOutput);
        } else {
          // Remove the last user input when model content is invalid
          curatedHistory.pop();
        }
      }
    }

    return curatedHistory;
  }

  /**
   * Check if content is valid
   * @param content - Content to validate
   * @returns True if content is valid
   */
  private isValidContent(content: Content): boolean {
    if (content.parts === undefined || content.parts.length === 0) {
      return false;
    }

    for (const part of content.parts) {
      if (part === undefined || Object.keys(part).length === 0) {
        return false;
      }
      if (!part.thought && part.text !== undefined && part.text === '') {
        return false;
      }
    }

    return true;
  }

  /**
   * Check if content is text content
   * @param content - Content to check
   * @returns True if content is text content
   */
  private isTextContent(
    content: Content | undefined,
  ): content is Content & { parts: [{ text: string }, ...Part[]] } {
    return !!(
      content &&
      content.role === 'model' &&
      content.parts &&
      content.parts.length > 0 &&
      typeof content.parts[0].text === 'string' &&
      content.parts[0].text !== ''
    );
  }

  /**
   * Check if content is thought content
   * @param content - Content to check
   * @returns True if content is thought content
   */
  private isThoughtContent(
    content: Content | undefined,
  ): content is Content & { parts: [{ thought: boolean }, ...Part[]] } {
    return !!(
      content &&
      content.role === 'model' &&
      content.parts &&
      content.parts.length > 0 &&
      typeof content.parts[0].thought === 'boolean' &&
      content.parts[0].thought === true
    );
  }

  /**
   * Find index after a fraction of total characters in history
   * @param history - History to analyze
   * @param fraction - Fraction (0-1) of content to find index for
   * @returns Index after the fraction point
   */
  private findIndexAfterFraction(history: Content[], fraction: number): number {
    if (fraction <= 0 || fraction >= 1) {
      throw new Error('Fraction must be between 0 and 1');
    }

    const contentLengths = history.map(
      (content) => JSON.stringify(content).length,
    );

    const totalCharacters = contentLengths.reduce(
      (sum, length) => sum + length,
      0,
    );
    const targetCharacters = totalCharacters * fraction;

    let charactersSoFar = 0;
    for (let i = 0; i < contentLengths.length; i++) {
      charactersSoFar += contentLengths[i];
      if (charactersSoFar >= targetCharacters) {
        return i;
      }
    }
    return contentLengths.length;
  }
}
