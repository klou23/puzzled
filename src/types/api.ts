/**
 * API Type Definitions
 * TypeScript interfaces for API requests and responses
 */

export interface HelpRequestResponse {
  joinUrl: string;
  startUrl?: string;
  meetingId: string;
}

export interface ApiError {
  error: string;
  detail?: string;
}
