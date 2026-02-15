"""
Email Service for Zoom Meeting Invitations
Handles SMTP connection and email delivery
"""

import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import aiosmtplib

logger = logging.getLogger(__name__)


class EmailService:
    """Email service for sending meeting invitations via SMTP"""

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        smtp_user: str,
        smtp_password: str,
        from_email: str,
        from_name: str = "TreeHacks26",
        use_tls: bool = True,
    ):
        """
        Initialize email service with SMTP configuration.

        Args:
            smtp_host: SMTP server hostname
            smtp_port: SMTP server port
            smtp_user: SMTP username
            smtp_password: SMTP password
            from_email: Email address to send from
            from_name: Display name for sender
            use_tls: Whether to use TLS encryption
        """
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password
        self.from_email = from_email
        self.from_name = from_name
        self.use_tls = use_tls

    async def send_meeting_invitation(
        self,
        recipient: str,
        meeting_data: dict,
    ) -> bool:
        """
        Send Zoom meeting invitation email.

        Args:
            recipient: Email address to send to
            meeting_data: Dict with join_url, meeting_id, password

        Returns:
            bool: True if sent successfully, False otherwise
        """
        try:
            # Create multipart message (HTML + plain text)
            message = MIMEMultipart("alternative")
            message["Subject"] = "[Hackathon Help] Your Zoom Meeting is Ready"
            message["From"] = f"{self.from_name} <{self.from_email}>"
            message["To"] = recipient

            # Generate content
            text_content = self._create_text_content(meeting_data)
            html_content = self._create_html_content(meeting_data)

            # Attach both versions (email clients will pick the best one)
            part1 = MIMEText(text_content, "plain")
            part2 = MIMEText(html_content, "html")
            message.attach(part1)
            message.attach(part2)

            # Send via SMTP
            await aiosmtplib.send(
                message,
                hostname=self.smtp_host,
                port=self.smtp_port,
                username=self.smtp_user,
                password=self.smtp_password,
                start_tls=self.use_tls,
                timeout=10.0,
            )

            logger.info(f"Successfully sent email to {recipient}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email to {recipient}: {str(e)}")
            return False

    def _create_text_content(self, meeting_data: dict) -> str:
        """
        Create plain text email content (fallback).

        Args:
            meeting_data: Dict with meeting details

        Returns:
            str: Plain text email content
        """
        password_line = (
            f"Password: {meeting_data.get('password')}"
            if meeting_data.get('password')
            else ""
        )

        return f"""
HACKATHON HELP REQUEST

A new help request has been submitted. Your Zoom meeting is ready.

MEETING DETAILS:
Meeting ID: {meeting_data['meeting_id']}
{password_line}
Duration: 40 minutes

JOIN MEETING:
{meeting_data['join_url']}

---
This is an automated message from the Hackathon Help System
Powered by TreeHacks26
        """.strip()

    def _create_html_content(self, meeting_data: dict) -> str:
        """
        Create HTML email content with professional styling.

        Args:
            meeting_data: Dict with meeting details

        Returns:
            str: HTML email content
        """
        password_section = ""
        if meeting_data.get('password'):
            password_section = f"<p><strong>Password:</strong> {meeting_data['password']}</p>"

        return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background: #2563eb; color: white; padding: 20px; text-align: center; border-radius: 8px 8px 0 0; }}
        .content {{ background: #f9fafb; padding: 20px; }}
        .meeting-details {{ background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 20px; margin: 20px 0; }}
        .button {{ display: inline-block; background: #2563eb; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; font-weight: bold; }}
        .footer {{ text-align: center; color: #6b7280; font-size: 12px; margin-top: 20px; padding: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1 style="margin: 0;">Your Hackathon Help Request</h1>
        </div>
        <div class="content">
            <p>Hello,</p>
            <p>A new help request has been submitted at the hackathon. A Zoom meeting has been created for you.</p>

            <div class="meeting-details">
                <h2 style="margin-top: 0;">Meeting Details</h2>
                <p><strong>Meeting ID:</strong> {meeting_data['meeting_id']}</p>
                {password_section}
                <p><strong>Duration:</strong> 40 minutes</p>
            </div>

            <div style="text-align: center; margin: 30px 0;">
                <a href="{meeting_data['join_url']}" class="button">Join Zoom Meeting</a>
            </div>

            <p style="font-size: 14px;">Or copy and paste this URL into your browser:</p>
            <p style="word-break: break-all; color: #2563eb; font-size: 12px;">{meeting_data['join_url']}</p>
        </div>
        <div class="footer">
            <p>This is an automated message from the Hackathon Help System</p>
            <p>Powered by TreeHacks26</p>
        </div>
    </div>
</body>
</html>
        """.strip()
