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
  GoogleGenAI,
} from '@google/genai';
import { createCodeAssistContentGenerator } from '../code_assist/codeAssist.js';
import { DEFAULT_GEMINI_MODEL } from '../config/models.js';
import { Config } from '../config/config.js';
import { getEffectiveModel } from './modelCheck.js';
import { UserTierId } from '../code_assist/types.js';
import { OpenAICompatibleGenerator } from './openaiCompatibleGenerator.js';

/**
 * Interface abstracting the core functionalities for generating content and counting tokens.
 */
export interface ContentGenerator {
  generateContent(
    request: GenerateContentParameters,
    userPromptId: string,
  ): Promise<GenerateContentResponse>;

  generateContentStream(
    request: GenerateContentParameters,
    userPromptId: string,
  ): Promise<AsyncGenerator<GenerateContentResponse>>;

  countTokens(request: CountTokensParameters): Promise<CountTokensResponse>;

  embedContent(request: EmbedContentParameters): Promise<EmbedContentResponse>;

  preprocess?(text?: string): string;

  userTier?: UserTierId;
}

export enum AuthType {
  LOGIN_WITH_GOOGLE = 'oauth-personal',
  USE_GEMINI = 'gemini-api-key',
  USE_VERTEX_AI = 'vertex-ai',
  CLOUD_SHELL = 'cloud-shell',
  USE_OLLAMA = 'ollama',
  USE_OPENAI = 'openai',
  USE_CUSTOM_OPENAI_COMPATIBLE = 'custom-openai-compatible',
}

export type ContentGeneratorConfig = {
  model: string;
  apiKey?: string;
  vertexai?: boolean;
  authType?: AuthType | undefined;
  proxy?: string | undefined;
  // New fields for external LLM providers
  baseUrl?: string;
  provider?: 'gemini' | 'ollama' | 'openai' | 'custom';
  headers?: Record<string, string>;
};

export function createContentGeneratorConfig(
  config: Config,
  authType: AuthType | undefined,
): ContentGeneratorConfig {
  const geminiApiKey = process.env.GEMINI_API_KEY || undefined;
  const googleApiKey = process.env.GOOGLE_API_KEY || undefined;
  const googleCloudProject = process.env.GOOGLE_CLOUD_PROJECT || undefined;
  const googleCloudLocation = process.env.GOOGLE_CLOUD_LOCATION || undefined;

  // External LLM provider environment variables
  const ollamaApiKey = process.env.OLLAMA_API_KEY || undefined;
  const ollamaBaseUrl = process.env.OLLAMA_BASE_URL || 'http://localhost:11434';
  const openaiApiKey = process.env.OPENAI_API_KEY || undefined;
  const openaiBaseUrl =
    process.env.OPENAI_BASE_URL || 'https://api.openai.com/v1';
  const customApiKey = process.env.CUSTOM_LLM_API_KEY || undefined;
  const customBaseUrl = process.env.CUSTOM_LLM_BASE_URL || undefined;

  // Use runtime model from config if available, otherwise fallback to parameter or default
  const effectiveModel = config.getModel() || DEFAULT_GEMINI_MODEL;

  const contentGeneratorConfig: ContentGeneratorConfig = {
    model: effectiveModel,
    authType,
    proxy: config?.getProxy(),
  };

  // If we are using Google auth or we are in Cloud Shell, there is nothing else to validate for now
  if (
    authType === AuthType.LOGIN_WITH_GOOGLE ||
    authType === AuthType.CLOUD_SHELL
  ) {
    contentGeneratorConfig.provider = 'gemini';
    return contentGeneratorConfig;
  }

  if (authType === AuthType.USE_GEMINI && geminiApiKey) {
    contentGeneratorConfig.apiKey = geminiApiKey;
    contentGeneratorConfig.vertexai = false;
    contentGeneratorConfig.provider = 'gemini';
    getEffectiveModel(
      contentGeneratorConfig.apiKey,
      contentGeneratorConfig.model,
      contentGeneratorConfig.proxy,
    );

    return contentGeneratorConfig;
  }

  if (
    authType === AuthType.USE_VERTEX_AI &&
    (googleApiKey || (googleCloudProject && googleCloudLocation))
  ) {
    contentGeneratorConfig.apiKey = googleApiKey;
    contentGeneratorConfig.vertexai = true;
    contentGeneratorConfig.provider = 'gemini';

    return contentGeneratorConfig;
  }

  // Ollama configuration
  if (authType === AuthType.USE_OLLAMA) {
    contentGeneratorConfig.apiKey = ollamaApiKey; // Ollama might not need API key
    contentGeneratorConfig.baseUrl = ollamaBaseUrl;
    contentGeneratorConfig.provider = 'ollama';
    contentGeneratorConfig.model = effectiveModel || 'deepseek-r1:latest';

    return contentGeneratorConfig;
  }

  // OpenAI configuration
  if (authType === AuthType.USE_OPENAI && openaiApiKey) {
    contentGeneratorConfig.apiKey = openaiApiKey;
    contentGeneratorConfig.baseUrl = openaiBaseUrl;
    contentGeneratorConfig.provider = 'openai';
    contentGeneratorConfig.model = effectiveModel || 'gpt-4';

    return contentGeneratorConfig;
  }

  // Custom OpenAI-compatible API configuration
  if (
    authType === AuthType.USE_CUSTOM_OPENAI_COMPATIBLE &&
    customApiKey &&
    customBaseUrl
  ) {
    contentGeneratorConfig.apiKey = customApiKey;
    contentGeneratorConfig.baseUrl = customBaseUrl;
    contentGeneratorConfig.provider = 'custom';

    return contentGeneratorConfig;
  }

  return contentGeneratorConfig;
}

export async function createContentGenerator(
  config: ContentGeneratorConfig,
  gcConfig: Config,
  sessionId?: string,
): Promise<ContentGenerator> {
  const version = process.env.CLI_VERSION || process.version;
  const httpOptions = {
    headers: {
      'User-Agent': `GeminiCLI/${version} (${process.platform}; ${process.arch})`,
    },
  };

  if (
    config.authType === AuthType.LOGIN_WITH_GOOGLE ||
    config.authType === AuthType.CLOUD_SHELL
  ) {
    return createCodeAssistContentGenerator(
      httpOptions,
      config.authType,
      gcConfig,
      sessionId,
    );
  }

  if (
    config.authType === AuthType.USE_GEMINI ||
    config.authType === AuthType.USE_VERTEX_AI
  ) {
    const googleGenAI = new GoogleGenAI({
      apiKey: config.apiKey === '' ? undefined : config.apiKey,
      vertexai: config.vertexai,
      httpOptions,
    });

    return googleGenAI.models;
  }

  // Handle external LLM providers
  if (
    config.authType === AuthType.USE_OLLAMA ||
    config.authType === AuthType.USE_OPENAI ||
    config.authType === AuthType.USE_CUSTOM_OPENAI_COMPATIBLE
  ) {
    if (!config.baseUrl) {
      throw new Error(`Base URL is required for ${config.authType}`);
    }

    return new OpenAICompatibleGenerator(
      config.baseUrl,
      config.model,
      config.apiKey,
      config.headers,
      gcConfig,
    );
  }

  throw new Error(
    `Error creating contentGenerator: Unsupported authType: ${config.authType}`,
  );
}
