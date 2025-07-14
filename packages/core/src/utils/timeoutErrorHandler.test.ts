/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, vi } from 'vitest';
import { isTimeoutError, handleTimeoutError, isGoogleDomainError, extractTargetAddress } from './timeoutErrorHandler.js';

describe('timeoutErrorHandler', () => {
  describe('isTimeoutError', () => {
    it('should detect ETIMEDOUT errors', () => {
      const error = new Error('connect ETIMEDOUT');
      (error as any).code = 'ETIMEDOUT';
      expect(isTimeoutError(error)).toBe(true);
    });

    it('should detect afterConnectMultiple errors', () => {
      const error = new Error('Something failed in afterConnectMultiple');
      expect(isTimeoutError(error)).toBe(true);
    });

    it('should detect internalConnectMultipleTimeout errors', () => {
      const error = new Error('Failed with internalConnectMultipleTimeout');
      expect(isTimeoutError(error)).toBe(true);
    });

    it('should detect various timeout patterns', () => {
      const timeoutErrors = [
        { message: 'Request timeout', code: undefined },
        { message: 'Connection timeout', code: undefined },
        { message: 'Socket timeout', code: undefined },
        { message: 'Connection reset by peer', code: 'ECONNRESET' },
        { message: 'getaddrinfo ENOTFOUND', code: 'ENOTFOUND' },
      ];

      timeoutErrors.forEach(({ message, code }) => {
        const error = new Error(message);
        if (code) {
          (error as any).code = code;
        }
        expect(isTimeoutError(error)).toBe(true);
      });
    });

    it('should not detect non-timeout errors', () => {
      const regularError = new Error('Authentication failed');
      expect(isTimeoutError(regularError)).toBe(false);
    });
  });

  describe('isGoogleDomainError', () => {
    it('should detect Google domain errors from message', () => {
      const googleErrors = [
        new Error('Failed to connect to googleapis.com'),
        new Error('Error from play.googleapis.com'),
        new Error('generativelanguage.googleapis.com returned 500'),
      ];

      googleErrors.forEach(error => {
        expect(isGoogleDomainError(error)).toBe(true);
      });
    });

    it('should detect Google domain errors from extracted target address', () => {
      const error = new Error('Connection failed');
      (error as any).hostname = 'generativelanguage.googleapis.com';
      expect(isGoogleDomainError(error)).toBe(true);
    });

    it('should not detect non-Google domain errors', () => {
      const regularError = new Error('Failed to connect to example.com');
      expect(isGoogleDomainError(regularError)).toBe(false);
    });
  });

  describe('handleTimeoutError', () => {
    it('should handle timeout errors in debug mode', () => {
      const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
      const mockConfig = {
        getDebugMode: () => true,
      } as any;

      const timeoutError = new Error('connect ETIMEDOUT');
      (timeoutError as any).code = 'ETIMEDOUT';

      const result = handleTimeoutError(timeoutError, mockConfig, 'test context');
      
      expect(result).toBe(true);
      expect(consoleSpy).toHaveBeenCalledWith(
        expect.stringContaining('[DEBUG] test context - Ignoring timeout error')
      );

      consoleSpy.mockRestore();
    });

    it('should not handle timeout errors when debug mode is disabled', () => {
      const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
      const mockConfig = {
        getDebugMode: () => false,
      } as any;

      const timeoutError = new Error('connect ETIMEDOUT');
      (timeoutError as any).code = 'ETIMEDOUT';

      const result = handleTimeoutError(timeoutError, mockConfig, 'test context');
      
      expect(result).toBe(false);
      expect(consoleSpy).not.toHaveBeenCalled();

      consoleSpy.mockRestore();
    });

    it('should not handle non-timeout errors', () => {
      const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
      const mockConfig = {
        getDebugMode: () => true,
      } as any;

      const regularError = new Error('Authentication failed');

      const result = handleTimeoutError(regularError, mockConfig, 'test context');
      
      expect(result).toBe(false);
      expect(consoleSpy).not.toHaveBeenCalled();

      consoleSpy.mockRestore();
    });

    it('should execute fallback action when provided', () => {
      const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
      const mockConfig = {
        getDebugMode: () => true,
      } as any;
      const fallbackSpy = vi.fn();

      const timeoutError = new Error('afterConnectMultiple failed');

      const result = handleTimeoutError(timeoutError, mockConfig, 'test context', fallbackSpy);
      
      expect(result).toBe(true);
      expect(fallbackSpy).toHaveBeenCalled();

      consoleSpy.mockRestore();
    });
  });

  describe('extractTargetAddress', () => {
    it('should extract address from error properties', () => {
      const error = new Error('Connection failed');
      (error as any).address = 'generativelanguage.googleapis.com';
      expect(extractTargetAddress(error)).toBe('generativelanguage.googleapis.com');
    });

    it('should extract hostname from error properties', () => {
      const error = new Error('Connection failed');
      (error as any).hostname = 'play.googleapis.com';
      expect(extractTargetAddress(error)).toBe('play.googleapis.com');
    });

    it('should extract URL from error message', () => {
      const error = new Error('Failed to connect to https://generativelanguage.googleapis.com/v1/models');
      expect(extractTargetAddress(error)).toBe('https://generativelanguage.googleapis.com/v1/models');
    });

    it('should extract hostname from connection error message', () => {
      const error = new Error('connect to generativelanguage.googleapis.com failed');
      expect(extractTargetAddress(error)).toBe('generativelanguage.googleapis.com');
    });

    it('should extract address from getaddrinfo error', () => {
      const error = new Error('getaddrinfo ENOTFOUND googleapis.com');
      expect(extractTargetAddress(error)).toBe('googleapis.com');
    });

    it('should return null for errors without address info', () => {
      const error = new Error('Generic error message');
      expect(extractTargetAddress(error)).toBe(null);
    });

    it('should return null for non-Error objects', () => {
      expect(extractTargetAddress('string error')).toBe(null);
      expect(extractTargetAddress(null)).toBe(null);
    });

    it('should extract IPv4 addresses from error messages', () => {
      const error = new Error('connect ETIMEDOUT 142.250.191.10:443');
      expect(extractTargetAddress(error)).toBe('142.250.191.10:443');
    });

    it('should extract IPv4 addresses without port', () => {
      const error = new Error('timeout connecting to 8.8.8.8');
      expect(extractTargetAddress(error)).toBe('8.8.8.8');
    });

    it('should extract IPv6 addresses from error messages', () => {
      const error = new Error('connect to 2001:4860:4860::8888:443 failed');
      expect(extractTargetAddress(error)).toBe('2001:4860:4860::8888:443');
    });

    it('should extract IPv6 addresses with brackets', () => {
      const error = new Error('timeout connecting to [2001:4860:4860::8888]:443');
      expect(extractTargetAddress(error)).toBe('[2001:4860:4860::8888]:443');
    });

    it('should handle afterConnectMultiple patterns with IP addresses', () => {
      const error = new Error('Error: afterConnectMultiple timeout to 172.217.14.138:443');
      expect(extractTargetAddress(error)).toBe('172.217.14.138:443');
    });
  });
});
