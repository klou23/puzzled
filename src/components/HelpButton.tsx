/**
 * HelpButton Component
 * Button to create and join a Zoom help meeting
 */

import { useState } from 'react';
import { createHelpMeeting, ApiError } from '../services/api';

export function HelpButton() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleHelpRequest = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await createHelpMeeting();

      // Open in new tab with security settings
      window.open(response.joinUrl, '_blank', 'noopener,noreferrer');

      // Optional: Log success for debugging
      console.log('Meeting created:', response.meetingId);
    } catch (err) {
      const apiError = err as ApiError;

      // User-friendly error messages based on status code
      if (apiError.statusCode === 503) {
        setError('Service temporarily unavailable. Please try again.');
      } else if (apiError.statusCode >= 500) {
        setError('Server error. Please contact support.');
      } else if (apiError.statusCode === 0) {
        setError('Network error. Check your connection and try again.');
      } else {
        setError(apiError.detail || 'Failed to create meeting. Please try again.');
      }

      console.error('Help request error:', apiError);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div style={{ marginBottom: '1rem' }}>
      <button
        onClick={handleHelpRequest}
        disabled={isLoading}
        style={{
          padding: '0.6rem 1.5rem',
          fontSize: '1rem',
          fontWeight: '500',
          color: 'white',
          backgroundColor: isLoading ? '#888' : '#646cff',
          border: 'none',
          borderRadius: '8px',
          cursor: isLoading ? 'not-allowed' : 'pointer',
          transition: 'background-color 0.25s',
        }}
        onMouseEnter={(e) => {
          if (!isLoading) {
            e.currentTarget.style.backgroundColor = '#535bf2';
          }
        }}
        onMouseLeave={(e) => {
          if (!isLoading) {
            e.currentTarget.style.backgroundColor = '#646cff';
          }
        }}
        aria-label="Request help"
      >
        {isLoading ? 'Creating Meeting...' : 'Get Help'}
      </button>

      {error && (
        <div
          role="alert"
          style={{
            marginTop: '0.5rem',
            padding: '0.5rem',
            color: '#ff6b6b',
            fontSize: '0.9rem',
            border: '1px solid #ff6b6b',
            borderRadius: '4px',
            backgroundColor: 'rgba(255, 107, 107, 0.1)',
          }}
        >
          {error}
        </div>
      )}
    </div>
  );
}
