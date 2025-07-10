/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { Box, Text, useInput } from 'ink';
import { Colors } from '../colors.js';
import { RadioButtonSelect } from './shared/RadioButtonSelect.js';
import { LoadedSettings, SettingScope } from '../../config/settings.js';
import { AuthType } from '@google/gemini-cli-core';
import { validateAuthMethod } from '../../config/auth.js';

interface AuthDialogProps {
  onSelect: (authMethod: AuthType | undefined, scope: SettingScope) => void;
  settings: LoadedSettings;
  initialErrorMessage?: string | null;
}

export function AuthDialog({
  onSelect,
  settings,
  initialErrorMessage,
}: AuthDialogProps): React.JSX.Element {
  const [errorMessage, setErrorMessage] = useState<string | null>(
    initialErrorMessage
      ? initialErrorMessage
      : process.env.GEMINI_API_KEY
        ? 'Existing API key detected (GEMINI_API_KEY). Select "Gemini API Key" option to use it.'
        : process.env.OPENAI_API_KEY
        ? 'Existing API key detected (OPENAI_API_KEY). Select "OpenAI" option to use it.'
        : process.env.CUSTOM_LLM_API_KEY && process.env.CUSTOM_LLM_BASE_URL
        ? 'Custom LLM configuration detected. Select "Custom OpenAI-Compatible" option to use it.'
        : process.env.OLLAMA_BASE_URL
        ? 'Ollama configuration detected. Select "Ollama (Local)" option to use it.'
        : null,
  );
  const items = [
    {
      label: 'Login with Google',
      value: AuthType.LOGIN_WITH_GOOGLE,
    },
    ...(process.env.CLOUD_SHELL === 'true'
      ? [
          {
            label: 'Use Cloud Shell user credentials',
            value: AuthType.CLOUD_SHELL,
          },
        ]
      : []),
    {
      label: 'Use Gemini API Key',
      value: AuthType.USE_GEMINI,
    },
    { label: 'Vertex AI', value: AuthType.USE_VERTEX_AI },
    { label: 'Ollama (Local)', value: AuthType.USE_OLLAMA },
    { label: 'OpenAI', value: AuthType.USE_OPENAI },
    { label: 'Custom OpenAI-Compatible', value: AuthType.USE_CUSTOM_OPENAI_COMPATIBLE },
  ];

  const initialAuthIndex = items.findIndex((item) => {
    if (settings.merged.selectedAuthType) {
      return item.value === settings.merged.selectedAuthType;
    }

    // Auto-detect based on environment variables
    if (process.env.GEMINI_API_KEY) {
      return item.value === AuthType.USE_GEMINI;
    }
    if (process.env.OPENAI_API_KEY) {
      return item.value === AuthType.USE_OPENAI;
    }
    if (process.env.CUSTOM_LLM_API_KEY && process.env.CUSTOM_LLM_BASE_URL) {
      return item.value === AuthType.USE_CUSTOM_OPENAI_COMPATIBLE;
    }
    // For Ollama, we'll default to it if the base URL is set or if it's running on localhost
    if (process.env.OLLAMA_BASE_URL) {
      return item.value === AuthType.USE_OLLAMA;
    }

    return item.value === AuthType.LOGIN_WITH_GOOGLE;
  });

  const handleAuthSelect = (authMethod: AuthType) => {
    const error = validateAuthMethod(authMethod);
    if (error) {
      setErrorMessage(error);
    } else {
      setErrorMessage(null);
      onSelect(authMethod, SettingScope.User);
    }
  };

  useInput((_input, key) => {
    if (key.escape) {
      // Prevent exit if there is an error message.
      // This means they user is not authenticated yet.
      if (errorMessage) {
        return;
      }
      if (settings.merged.selectedAuthType === undefined) {
        // Prevent exiting if no auth method is set
        setErrorMessage(
          'You must select an auth method to proceed. Press Ctrl+C twice to exit.',
        );
        return;
      }
      onSelect(undefined, SettingScope.User);
    }
  });

  return (
    <Box
      borderStyle="round"
      borderColor={Colors.Gray}
      flexDirection="column"
      padding={1}
      width={process.platform === 'win32' ? '48%' : '98%'}
    >
      <Text bold>Get started</Text>
      <Box marginTop={1}>
        <Text>How would you like to authenticate for this project?</Text>
      </Box>
      <Box marginTop={1}>
        <RadioButtonSelect
          items={items}
          initialIndex={initialAuthIndex}
          onSelect={handleAuthSelect}
          isFocused={true}
        />
      </Box>
      {errorMessage && (
        <Box marginTop={1}>
          <Text color={Colors.AccentRed}>{errorMessage}</Text>
        </Box>
      )}
      <Box marginTop={1}>
        <Text color={Colors.Gray}>(Use Enter to select)</Text>
      </Box>
      <Box marginTop={1}>
        <Text>Terms of Services and Privacy Notice for Gemini CLI</Text>
      </Box>
      <Box marginTop={1}>
        <Text color={Colors.AccentBlue}>
          {
            'https://github.com/google-gemini/gemini-cli/blob/main/docs/tos-privacy.md'
          }
        </Text>
      </Box>
    </Box>
  );
}
