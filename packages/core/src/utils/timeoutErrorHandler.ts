/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { Config } from '../config/config.js';

/**
 * Check if an error is a timeout-related error that should be handled gracefully in debug mode
 */
export function isTimeoutError(error: Error | unknown): boolean {
  if (!(error instanceof Error)) {
    return false;
  }

  const errorMessage = error.message?.toLowerCase() || '';
  const errorName = error.name?.toLowerCase() || '';
  const errorStack = error.stack?.toLowerCase() || '';
  
  // Check error code property first (most reliable)
  const errorCode = (error as any).code?.toLowerCase();
  const timeoutCodes = [
    'etimedout',
    'econnreset',
    'enotfound',
    'enetunreach',
    'ehostunreach',
    'econnrefused',
    'eproto',
    'econnaborted',
    'emfile',
    'enfile',
    'eaddrnotavail'
  ];
  
  if (errorCode && timeoutCodes.includes(errorCode)) {
    return true;
  }
  
  // Specific timeout patterns (exact matches to avoid false positives)
  const specificTimeoutPatterns = [
    'afterconnectmultiple',
    'internalconnectmultipletimeout',
    'connect timeout',
    'request timeout',
    'socket timeout',
    'network timeout',
    'connection timeout',
    'timeout error',
    'timed out',
    'timeout expired'
  ];
  
  // Check for specific timeout patterns
  return specificTimeoutPatterns.some(pattern => 
    errorMessage.includes(pattern) || 
    errorName.includes(pattern) ||
    errorStack.includes(pattern)
  );
}

/**
 * Extract target address/URL from error if available
 * This can be a domain name, IPv4 address, IPv6 address, or full URL
 */
export function extractTargetAddress(error: Error | unknown): string | null {
  if (!(error instanceof Error)) {
    return null;
  }

  const errorMessage = error.message || '';
  const errorStack = error.stack || '';
  
  // Check for common URL and address patterns in error messages
  const addressPatterns = [
    // Full URLs (http/https)
    /https?:\/\/[^\s]+/gi,
    // IPv4 addresses (with optional port) - more specific pattern
    /(?:connect (?:to )?|connecting to |request to |failed to reach |timeout.*?to |address:\s*|host:\s*|connect\s+ETIMEDOUT\s+)(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?::\d+)?)/gi,
    // IPv6 addresses (basic pattern, with optional port)
    /(?:connect (?:to )?|connecting to |request to |failed to reach |timeout.*?to |address:\s*|host:\s*)(\[?[0-9a-f:]+::[0-9a-f:]*\]?(?::\d+)?)/gi,
    // Domain names (including subdomains, with optional port) - more specific
    /(?:connect (?:to )?|connecting to |request to |failed to reach |timeout.*?to |address:\s*|host:\s*|getaddrinfo\s+[A-Z_]+\s+)([a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?::\d+)?)/gi,
    // afterConnectMultiple patterns
    /afterConnectMultiple.*?(?:to|timeout to)\s+([^\s]+)/gi,
    // Generic address patterns in error messages (last resort)
    /address:\s*([^\s]+)/gi,
    // Host patterns (last resort)
    /host:\s*([^\s]+)/gi
  ];

  // Try to extract from error properties first
  const errorObj = error as any;
  if (errorObj.address) {
    return errorObj.address;
  }
  if (errorObj.hostname) {
    return errorObj.hostname;
  }
  if (errorObj.host) {
    return errorObj.host;
  }
  if (errorObj.config?.url) {
    return errorObj.config.url;
  }
  if (errorObj.request?.host) {
    return errorObj.request.host;
  }

  // Try to extract from error message and stack
  const fullText = `${errorMessage} ${errorStack}`;
  
  for (const pattern of addressPatterns) {
    pattern.lastIndex = 0; // Reset regex state
    const match = pattern.exec(fullText);
    if (match) {
      // If pattern has capture groups, use the first capture group
      if (match[1]) {
        return match[1].trim();
      }
      // Otherwise use the full match
      return match[0].trim();
    }
  }

  return null;
}

/**
 * Check if an error is related to Google domains/services
 */
export function isGoogleDomainError(error: Error | unknown): boolean {
  if (!(error instanceof Error)) {
    return false;
  }

  const errorMessage = error.message?.toLowerCase() || '';
  const errorStack = error.stack?.toLowerCase() || '';
  const targetAddress = extractTargetAddress(error)?.toLowerCase() || '';
  
  const googleDomains = [
    'googleapis.com',
    'play.googleapis.com',
    'generativelanguage.googleapis.com',
    'cloudcode-pa.googleapis.com',
    'google.com',
    'gstatic.com'
  ];
  
  return googleDomains.some(domain => 
    errorMessage.includes(domain) || 
    errorStack.includes(domain) ||
    targetAddress.includes(domain)
  );
}

/**
 * Handle timeout errors gracefully in debug mode
 * @param error The error to handle
 * @param config Configuration object
 * @param context Context description for logging
 * @param fallbackAction Optional fallback action to execute
 * @returns true if error was handled (should not be re-thrown), false otherwise
 */
export function handleTimeoutError(
  error: Error | unknown,
  config: Config | undefined,
  context: string,
  fallbackAction?: () => void
): boolean {
  if (!config?.getDebugMode()) {
    return false; // Don't handle in non-debug mode
  }
  
  if (!isTimeoutError(error)) {
    return false; // Not a timeout error
  }
  
  // Log the ignored timeout error in debug mode
  const errorMessage = error instanceof Error ? error.message : String(error);
  const isGoogleRelated = isGoogleDomainError(error);
  const targetAddress = extractTargetAddress(error);
  
  let logMessage = `[DEBUG] ${context} - Ignoring timeout error`;
  
  if (targetAddress) {
    logMessage += ` to ${targetAddress}`;
  }
  
  if (isGoogleRelated) {
    logMessage += ' (Google domain)';
  }
  
  logMessage += `: ${errorMessage}`;
  
  console.log(logMessage);
  
  // Execute fallback action if provided
  if (fallbackAction) {
    try {
      fallbackAction();
    } catch (fallbackError) {
      console.log(`[DEBUG] ${context} - Fallback action failed:`, fallbackError);
    }
  }
  
  return true; // Error was handled
}

/**
 * Wrapper for async functions that may encounter timeout errors
 * @param asyncFn The async function to wrap
 * @param config Configuration object
 * @param context Context description for logging
 * @param fallbackResult Optional fallback result to return on timeout
 */
export async function withTimeoutErrorHandling<T>(
  asyncFn: () => Promise<T>,
  config: Config | undefined,
  context: string,
  fallbackResult?: T
): Promise<T | undefined> {
  try {
    return await asyncFn();
  } catch (error) {
    const wasHandled = handleTimeoutError(
      error,
      config,
      context,
      fallbackResult ? () => console.log(`[DEBUG] Using fallback result for ${context}`) : undefined
    );
    
    if (wasHandled) {
      return fallbackResult;
    }
    
    // Re-throw if not handled
    throw error;
  }
}
