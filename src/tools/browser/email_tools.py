import base64
import time
import datetime
from email.message import EmailMessage
from googleapiclient.discovery import build
from src.tools.google_auth import get_google_credentials

def get_unread_emails(max_results: int = 5):
    """
    Fetches the latest unread emails from the user's Gmail inbox.
    Returns a string summarizing the emails.
    """
    try:
        creds = get_google_credentials()
        service = build('gmail', 'v1', credentials=creds)

        # Call the Gmail API to fetch UNREAD messages
        results = service.users().messages().list(
            userId='me', 
            labelIds=['INBOX', 'UNREAD'], 
            maxResults=max_results
        ).execute()
        
        messages = results.get('messages', [])

        if not messages:
            return "No new unread emails found."

        email_summaries = []
        for msg in messages:
            msg_data = service.users().messages().get(userId='me', id=msg['id'], format='full').execute()
            payload = msg_data.get('payload', {})
            headers = payload.get('headers', [])
            
            subject = "No Subject"
            sender = "Unknown Sender"
            
            for header in headers:
                if header['name'] == 'Subject':
                    subject = header['value']
                if header['name'] == 'From':
                    sender = header['value']
            
            # Simple body extraction
            body = "No body content"
            if 'parts' in payload:
                for part in payload['parts']:
                    if part['mimeType'] == 'text/plain':
                        data = part['body'].get('data')
                        if data:
                            body = base64.urlsafe_b64decode(data).decode('utf-8')
                            break
            elif 'body' in payload and 'data' in payload['body']:
                data = payload['body']['data']
                if data:
                    body = base64.urlsafe_b64decode(data).decode('utf-8')

            email_summaries.append(f"From: {sender}\nSubject: {subject}\nSnippet: {msg_data.get('snippet', '')}\nBody: {body[:200]}...")
            
        return "\n\n".join(email_summaries)

    except Exception as e:
        return f"An error occurred fetching emails: {e}"

def send_email(to: str, subject: str, body: str):
    """
    Sends an email using the user's Gmail account.
    """
    try:
        creds = get_google_credentials()
        service = build('gmail', 'v1', credentials=creds)

        message = EmailMessage()
        message.set_content(body)
        message['To'] = to
        message['From'] = 'me'
        message['Subject'] = subject

        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

        create_message = {
            'raw': encoded_message
        }

        # pylint: disable=E1101
        send_message = (service.users().messages().send(userId="me", body=create_message).execute())
        
        return f"Email sent successfully. Message Id: {send_message['id']}"
    except Exception as e:
        return f"An error occurred sending the email: {e}"

def wait_for_new_email(poll_interval_seconds: int = 15):
    """
    Continuously polls the Gmail API until a new email arrives in the inbox.
    Blocks execution until a new email is detected, then returns its summary.
    """
    try:
        creds = get_google_credentials()
        service = build('gmail', 'v1', credentials=creds)

        profile = service.users().getProfile(userId='me').execute()
        start_history_id = profile['historyId']

        print(f"Waiting for new emails (polling every {poll_interval_seconds}s)...")
        while True:
            time.sleep(poll_interval_seconds)
            try:
                history = service.users().history().list(
                    userId='me', 
                    startHistoryId=start_history_id, 
                    historyTypes=['messageAdded']
                ).execute()

                if 'history' in history:
                    # New message arrived!
                    # Return the single latest unread email to avoid confusing the LLM
                    return get_unread_emails(max_results=1)
                
                start_history_id = history.get('historyId', start_history_id)
            except Exception as e:
                print(f"Polling error: {e}. Retrying...")
    except Exception as e:
        return f"An error occurred waiting for emails: {e}"

def get_upcoming_events(max_results: int = 10):
    """
    Fetches upcoming events from the user's Google Calendar.
    Returns a string summarizing the events.
    """
    try:
        creds = get_google_credentials()
        service = build('calendar', 'v3', credentials=creds)

        # Call the Calendar API
        now = datetime.datetime.utcnow().isoformat() + 'Z'  # 'Z' indicates UTC time
        events_result = service.events().list(
            calendarId='primary', 
            timeMin=now,
            maxResults=max_results, 
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        
        events = events_result.get('items', [])

        if not events:
            return 'No upcoming events found.'

        event_summaries = []
        for event in events:
            start = event['start'].get('dateTime', event['start'].get('date'))
            summary = event.get('summary', 'No Title')
            event_summaries.append(f"{start} - {summary}")
            
        return "\n".join(event_summaries)
        
    except Exception as e:
        return f"An error occurred fetching events: {e}"

def create_event(summary: str, description: str, start_time_iso: str, end_time_iso: str):
    """
    Creates an event in the user's primary Google Calendar.
    start_time_iso and end_time_iso should be in ISO 8601 format (e.g., '2023-10-25T10:00:00-07:00').
    """
    try:
        creds = get_google_credentials()
        service = build('calendar', 'v3', credentials=creds)

        event = {
            'summary': summary,
            'description': description,
            'start': {
                'dateTime': start_time_iso,
            },
            'end': {
                'dateTime': end_time_iso,
            },
        }

        event = service.events().insert(calendarId='primary', body=event).execute()
        return f"Event created: {event.get('htmlLink')}"
        
    except Exception as e:
        return f"An error occurred creating the event: {e}"
