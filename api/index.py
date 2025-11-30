from flask import Flask, request, jsonify
import os
import datetime
import json
from typing import List, Dict

# Google Auth Imports
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

# OpenAI Direct Client (no LangChain)
from openai import OpenAI

# MongoDB Imports
from pymongo import MongoClient

# =============================================================================
# CONFIGURATION
# =============================================================================

SCOPES = ['https://www.googleapis.com/auth/calendar']
# Fixed: Use the correct model name without "models/" prefix
# Options: "gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"
MODEL_NAME = "gpt-4o"

# Prompt
AGENT_SYSTEM_PROMPT = """You are an expert Executive Assistant and Calendar Conflict Resolver.

Responsibilities:
1. Manage the user's Google Calendar.
2. Detect conflicts (overlapping events).
3. Propose and execute solutions (rescheduling).

Guidelines:
- ALWAYS call `get_current_time_and_events` before proposing time slots or editing events.
- When resolving a conflict, look for the first available empty slot that fits the duration.
- If the user references a meeting by name, list events to find the corresponding ID before acting.
- Communicate clearly, be polite, and keep responses action-focused.
- The user may append the current date in their request; rely on it for temporal context."""

app = Flask(__name__)

# =============================================================================
# MONGODB CONNECTION
# =============================================================================

MONGO_URI = os.environ.get('MONGO_URI')

if not MONGO_URI:
    print("WARNING: MONGO_URI not set. Database operations will fail.")
    client = None
    tokens_collection = None
else:
    try:
        client = MongoClient(MONGO_URI)
        db = client.get_database('calendar_agent')
        tokens_collection = db.tokens
        print("Connected to MongoDB successfully.")
    except Exception as e:
        print(f"Failed to connect to MongoDB: {e}")
        client = None
        tokens_collection = None

DEFAULT_USER_ID = "main_user"

# =============================================================================
# HELPER FUNCTIONS: DATABASE
# =============================================================================

def save_token_to_db(creds):
    """Saves the credentials object to MongoDB."""
    if tokens_collection is None:
        print("Error: Database not connected.")
        return False
    
    creds_dict = json.loads(creds.to_json())
    
    tokens_collection.update_one(
        {"user_id": DEFAULT_USER_ID},
        {"$set": {"token_data": creds_dict, "updated_at": datetime.datetime.utcnow()}},
        upsert=True
    )
    print("Token saved to MongoDB.")
    return True

def load_token_from_db():
    """Loads credentials from MongoDB."""
    if tokens_collection is None:
        return None
        
    user_doc = tokens_collection.find_one({"user_id": DEFAULT_USER_ID})
    
    if user_doc and "token_data" in user_doc:
        return Credentials.from_authorized_user_info(user_doc["token_data"], SCOPES)
    
    return None

# =============================================================================
# GOOGLE CALENDAR SERVICE
# =============================================================================

class GoogleCalendarService:
    def __init__(self, credentials=None):
        self.creds = credentials
        self.service = None

        if self.creds:
            if not self.creds.valid:
                if self.creds.expired and self.creds.refresh_token:
                    try:
                        print("Token expired. Refreshing...")
                        self.creds.refresh(Request())
                        save_token_to_db(self.creds) 
                    except Exception as e:
                        print(f"Error refreshing token: {e}")
                        self.creds = None
        
        if self.creds:
            self.service = build('calendar', 'v3', credentials=self.creds)

    def list_events(self, days: int = 7) -> str:
        """List events for the next X days."""
        if not self.service:
            return "Calendar service not initialized"
        
        now = datetime.datetime.utcnow().isoformat() + 'Z'
        end_date = (datetime.datetime.utcnow() + datetime.timedelta(days=days)).isoformat() + 'Z'
        
        try:
            events_result = self.service.events().list(
                calendarId='primary', timeMin=now, timeMax=end_date,
                singleEvents=True, orderBy='startTime').execute()
            events = events_result.get('items', [])

            if not events:
                return "No upcoming events found."
            
            result_str = f"--- Calendar Events for the next {days} days ---\n"
            for event in events:
                start = event['start'].get('dateTime', event['start'].get('date'))
                event_id = event['id']
                summary = event.get('summary', 'No Title')
                result_str += f"- [ID: {event_id}] {start}: {summary}\n"
            
            return result_str
        except Exception as e:
            return f"API Error: {str(e)}"

    def create_event(self, summary: str, start_iso: str, end_iso: str, description: str = ""):
        """Create a new event."""
        if not self.service:
            return "Calendar service not initialized"
        
        event = {
            'summary': summary,
            'description': description,
            'start': {'dateTime': start_iso, 'timeZone': 'UTC'},
            'end': {'dateTime': end_iso, 'timeZone': 'UTC'},
        }
        try:
            event = self.service.events().insert(calendarId='primary', body=event).execute()
            return f"Event created: {event.get('htmlLink')}"
        except Exception as e:
            return f"Failed to create event: {str(e)}"

    def update_event_time(self, event_id: str, new_start_iso: str, new_end_iso: str):
        """Reschedule an existing event."""
        if not self.service:
            return "Calendar service not initialized"
        
        try:
            event = self.service.events().get(calendarId='primary', eventId=event_id).execute()
            event['start']['dateTime'] = new_start_iso
            event['end']['dateTime'] = new_end_iso
            
            updated_event = self.service.events().update(
                calendarId='primary', eventId=event_id, body=event).execute()
            
            return f"Event updated successfully. New link: {updated_event.get('htmlLink')}"
        except Exception as e:
            return f"Failed to update event: {str(e)}"

calendar_service = None

# =============================================================================
# LANGCHAIN TOOLS
# =============================================================================

def get_current_time_and_events() -> str:
    """
    Get the current system time (UTC) and list the user's upcoming calendar events.
    ALWAYS call this first to understand the schedule context before making changes.
    """
    global calendar_service
    if not calendar_service or not calendar_service.service:
        return "Authentication required. Please visit /api/auth"
    
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    events = calendar_service.list_events(days=5)
    return f"Current Time (UTC): {now}\n\n{events}"

def schedule_event(summary: str, start_time: str, duration_minutes: int, description: str = "") -> str:
    """
    Schedule a new event.
    start_time must be in ISO format (YYYY-MM-DDTHH:MM:SS).
    """
    global calendar_service
    if not calendar_service:
        return "Calendar service not initialized"
    
    try:
        dt_start = datetime.datetime.fromisoformat(start_time)
        dt_end = dt_start + datetime.timedelta(minutes=duration_minutes)
        return calendar_service.create_event(summary, start_time, dt_end.isoformat(), description)
    except ValueError:
        return "Error: start_time must be in ISO format (e.g., 2024-11-29T15:00:00)"

def resolve_conflict_reschedule(event_id: str, new_start_time: str) -> str:
    """
    Reschedule an existing event (identified by ID) to a new start time.
    Preserves duration. new_start_time must be in ISO format.
    """
    global calendar_service
    if not calendar_service:
        return "Calendar service not initialized"
    
    try:
        event = calendar_service.service.events().get(calendarId='primary', eventId=event_id).execute()
        
        old_start = event['start'].get('dateTime')
        old_end = event['end'].get('dateTime')
        
        if not old_start or not old_end:
            return "Cannot reschedule all-day events with this tool."

        dt_start_old = datetime.datetime.fromisoformat(old_start)
        dt_end_old = datetime.datetime.fromisoformat(old_end)
        duration = dt_end_old - dt_start_old
        
        dt_new_start = datetime.datetime.fromisoformat(new_start_time)
        dt_new_end = dt_new_start + duration
        
        return calendar_service.update_event_time(event_id, new_start_time, dt_new_end.isoformat())

    except Exception as e:
        return f"Error rescheduling: {str(e)}"

# =============================================================================
# OPENAI CLIENT (NO LANGCHAIN)
# =============================================================================

def get_openai_client() -> OpenAI | None:
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        return None
    return OpenAI(api_key=api_key)

def run_assistant(messages: list) -> str:
    client = get_openai_client()
    if not client:
        return "OPENAI_API_KEY not set"
    chat_messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
    for m in messages:
        role = m.get('role')
        content = m.get('content')
        if role and content:
            chat_messages.append({"role": role, "content": content})
    try:
        resp = client.chat.completions.create(model=MODEL_NAME, messages=chat_messages, temperature=0)
        return resp.choices[0].message.content if resp.choices else "(no response)"
    except Exception as e:
        return f"Model error: {e}"

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Google Calendar Conflict Resolver API (MongoDB Backed)",
        "endpoints": {
            "/api/auth": "GET - Get OAuth URL",
            "/api/callback": "POST - Handle OAuth callback & Save to Mongo",
            "/api/chat": "POST - Chat with agent (Uses Mongo Token)"
        }
    })

@app.route('/api/auth', methods=['GET'])
def auth():
    """Generate OAuth URL for user authentication"""
    try:
        credentials_json = os.environ.get('GOOGLE_CREDENTIALS')
        if not credentials_json:
            return jsonify({"error": "GOOGLE_CREDENTIALS not set"}), 500
        
        credentials_dict = json.loads(credentials_json)
        
        flow = Flow.from_client_config(
            credentials_dict,
            scopes=SCOPES,
            redirect_uri=os.environ.get('REDIRECT_URI', 'http://localhost:3000/callback')
        )
        
        authorization_url, state = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true'
        )
        
        return jsonify({
            "auth_url": authorization_url,
            "state": state
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/callback', methods=['POST'])
def callback():
    """Handle OAuth callback, exchange code for token, and SAVE TO MONGODB"""
    try:
        data = request.json
        code = data.get('code')
        
        if not code:
            return jsonify({"error": "Authorization code required"}), 400
        
        credentials_json = os.environ.get('GOOGLE_CREDENTIALS')
        if not credentials_json:
            return jsonify({"error": "GOOGLE_CREDENTIALS not set"}), 500
            
        credentials_dict = json.loads(credentials_json)
        
        flow = Flow.from_client_config(
            credentials_dict,
            scopes=SCOPES,
            redirect_uri=os.environ.get('REDIRECT_URI', 'http://localhost:3000/callback')
        )
        
        flow.fetch_token(code=code)
        credentials = flow.credentials
        
        success = save_token_to_db(credentials)
        if not success:
             return jsonify({"error": "Failed to save token to database"}), 500
        
        return jsonify({
            "message": "Authentication successful. Token saved to MongoDB."
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Process user message using Token from MONGODB"""
    global calendar_service
    
    try:
        data = request.json
        user_message = data.get('message')
        conversation_history = data.get('history', [])
        
        if not user_message:
            return jsonify({"error": "Message required"}), 400

        # Check for OpenAI API key
        if not os.environ.get('OPENAI_API_KEY'):
            return jsonify({"error": "OPENAI_API_KEY not set"}), 500
        
        # Load token from MongoDB
        creds = load_token_from_db()
        
        if not creds:
            return jsonify({
                "error": "User not authenticated. Please go to /api/auth first.", 
                "auth_required": True
            }), 401
            
        calendar_service = GoogleCalendarService(credentials=creds)
        
        if not calendar_service.service:
             return jsonify({"error": "Failed to initialize Calendar service with stored token"}), 401

        # Prepare messages for OpenAI
        messages = list(conversation_history or [])
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        messages.append({"role": "user", "content": f"{user_message}\n\n(Current date: {current_date})"})

        response_text = run_assistant(messages)

        return jsonify({
            "response": response_text,
            "success": True
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)