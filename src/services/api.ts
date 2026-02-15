/**
 * API Client Service
 * Handles communication with the backend API
 */

import type { HelpRequestResponse } from '../types/api';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

/**
 * Custom API Error class for structured error handling
 */
export class ApiError extends Error {
  statusCode: number;
  detail?: string;

  constructor(
    message: string,
    statusCode: number,
    detail?: string
  ) {
    super(message);
    this.statusCode = statusCode;
    this.detail = detail;
    this.name = 'ApiError';
  }
}

/**
 * Create a help meeting via the backend API
 *
 * @returns Promise with meeting URLs and ID
 * @throws ApiError if request fails
 */
export async function createHelpMeeting(): Promise<HelpRequestResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/help`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new ApiError(
        'Failed to create help meeting',
        response.status,
        errorData.detail
      );
    }

    return await response.json();
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }
    // Network error or other unexpected error
    throw new ApiError(
      'Network error: Unable to reach the server',
      0,
      error instanceof Error ? error.message : 'Unknown error'
    );
  }
}
