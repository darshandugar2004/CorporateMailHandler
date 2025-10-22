import os.path
import base64
import json
import time
from email.mime.text import MIMEText
import re

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
# from google import drive

# --- IMPORT THE LANGGRAPH WORKFLOW ---
from main_graph import run_workflow  # Import the helper function
# from config import configg

# === CONFIGURATION ===
SCOPES = ['https://www.googleapis.com/auth/gmail.modify']
CREDENTIALS_FILE = 'credentials.json'
TOKEN_FILE = 'token.json'
POLLING_INTERVAL_SECONDS = 30 

# --- (get_gmail_service, mark_as_read, create_message, send_email functions remain the same) ---
def get_gmail_service():
    """Authenticates with the Gmail API and returns a service object."""
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        
        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())
            
    try:
        service = build('gmail', 'v1', credentials=creds)
        return service
    except HttpError as error:
        print(f'An error occurred: {error}')
        return None

def mark_as_read(service, msg_id):
    """Marks an email as read by removing the 'UNREAD' label."""
    try:
        service.users().messages().modify(
            userId='me', 
            id=msg_id, 
            body={'removeLabelIds': ['UNREAD']}
        ).execute()
    except HttpError as error:
        print(f'An error occurred while marking as read: {error}')

def create_message(sender, to, subject, message_text):
    message = MIMEText(message_text)
    message['to'] = to
    message['from'] = sender
    message['subject'] = subject
    raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
    return {'raw': raw_message}

def send_email(service, to, subject, body):
    """Sends an email."""
    try:
        user_profile = service.users().getProfile(userId='me').execute()
        sender_email = user_profile['emailAddress']
        
        message_body = create_message(sender_email, to, subject, body)
        
        message = service.users().messages().send(
            userId='me',
            body=message_body
        ).execute()
        
        print(f"Reply sent successfully. Message Id: {message['id']}")
        return message

    except HttpError as error:
        print(f'An error occurred while sending: {error}')
        return None

def get_unread_emails(service, start_timestamp):
    """Lists unread emails received AFTER the start_timestamp."""
    try:
        result = service.users().messages().list(
            userId='me', 
            q=f"is:unread in:inbox after:{start_timestamp}"
        ).execute()
        
        messages = result.get('messages', [])
        
        if not messages:
            return []

        email_list = []
        for msg in messages:
            msg_data = service.users().messages().get(
                userId='me', 
                id=msg['id'],
                format='full'
            ).execute()
            
            payload = msg_data['payload']
            headers = payload['headers']
            
            email_info = {
                'id': msg['id'],
                'threadId': msg_data['threadId'],
                'snippet': msg_data['snippet'],
                'subject': 'No Subject',
                'sender_full': 'Unknown Sender',
                'sender_email': 'unknown@example.com',
                'content': ''
            }

            for header in headers:
                name = header['name'].lower()
                if name == 'subject':
                    email_info['subject'] = header['value']
                if name == 'from':
                    email_info['sender_full'] = header['value']
                    match = re.search(r'<(.*?)>', header.get('value', ''))
                    if match:
                        email_info['sender_email'] = match.group(1)
                    else:
                        email_info['sender_email'] = header.get('value', 'unknown@example.com')
            
            body_data = ""
            if 'parts' in payload:
                for part in payload['parts']:
                    if part['mimeType'] == 'text/plain':
                        body_data = part['body']['data']
                        break
            elif 'body' in payload:
                body_data = payload['body']['data']
            
            if body_data:
                email_info['content'] = base64.urlsafe_b64decode(
                    body_data.encode('ASCII')
                ).decode('utf-8')
            
            email_list.append(email_info)
            
        return email_list

    except HttpError as error:
        print(f'An error occurred: {error}')
        return []

if __name__ == "__main__":

    # configg()

    '''
    # 1. Create an instance of the generator.
    #    (This is the slow part that loads the model ONCE.)
    email_bot = EmailGenerator(base_model_id=BASE_MODEL_ID, adapter_path=ADAPTER_PATH)

    # 2. Define the email requirements.
    email_intent = "Financial Performance Summary"
    email_details = ("Summarize the key findings from the Q3 financial report. "
                     "Mention the 15% revenue growth, the successful launch of "
                     "Project Phoenix, and the new projection of a 12% profit margin for Q4. "
                     "The email should be addressed to all department heads.")

    # 3. Generate the reply.
    #    (This part is fast and can be called repeatedly.)
    print("--- Generating Email ---")
    generated_email = email_bot.generate(intent=email_intent, details=email_details)

    # 4. Print the result.
    print("\n--- Generated Email ---\n")
    print(generated_email)

    # Example of a second, different request (fast because the model is already loaded)
    print("\n\n--- Generating Second Email ---")
    generated_email_2 = email_bot.generate(
        intent="Merger Announcement",
        details=("Announce the successful merger with Innovate Corp. "
                 "Emphasize the strategic benefits and the expected synergies. "
                 "A town hall meeting is scheduled for next Friday at 10 AM.")
    )
    print("\n--- Generated Email ---\n")
    print(generated_email_2)
    '''

    print("Starting mail attender service...")
    service = get_gmail_service()
    
    if not service:
        print("Failed to authenticate with Gmail. Exiting.")
    else:
        start_time_unix = int(time.time()) 
        print(f"Service running. Ignoring emails received before startup.")
        print(f"Checking for new mail every {POLLING_INTERVAL_SECONDS} seconds...")
        
        while True:
            try:
                emails = get_unread_emails(service, start_time_unix)
                
                if emails:
                    print(f"\n--- Found {len(emails)} new email(s)! ---")
                    
                    for email in emails:
                        # 1. Prepare data
                        email_data = {
                            "sender": email['sender_email'],
                            "subject": email['subject'],
                            "content": email['content'].strip()
                        }
                        
                        print("--- Calling LangGraph Workflow ---")
                        print(f"Processing email from: {email_data['sender']}")
                        
                        # 2. Run the LangGraph workflow
                        reply = run_workflow(
                            email_content=email_data['content'],
                            sender_email=email_data['sender'],
                            subject=email_data['subject']
                        )
                        
                        print("--- Workflow Generated Reply ---")
                        
                        # 3. Send the AI-generated reply
                        send_email(
                            service, 
                            email_data['sender'],
                            reply['reply_subject'], 
                            reply['reply_body']
                        )
                        
                        # 4. Mark original email as read
                        mark_as_read(service, email['id'])
                        print(f"Processed and marked email {email['id']} as read.")
                    
                    print(f"\nWaiting for {POLLING_INTERVAL_SECONDS} seconds...")

                else:
                    print(f". (No new mail. Waiting {POLLING_INTERVAL_SECONDS}s)", end="", flush=True)

                time.sleep(POLLING_INTERVAL_SECONDS)
            
            except KeyboardInterrupt:
                print("\nStopping service.")
                break
            except Exception as e:
                print(f"\nAn error occurred in the main loop: {e}")
                print(f"Restarting loop in {POLLING_INTERVAL_SECONDS} seconds...")
                time.sleep(POLLING_INTERVAL_SECONDS)