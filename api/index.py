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

# LangChain Imports (Gemini)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# MongoDB Imports
from pymongo import MongoClient

# =============================================================================
# CONFIGURATION
# =============================================================================

SCOPES = ['https://www.googleapis.com/auth/calendar']
# Use a GA-supported Gemini model identifier
# Common options: "gemini-1.5-flash-latest" or "gemini-1.5-pro-latest"
MODEL_NAME = "gemini-1.5-flash-latest"

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

# Ideally, ensure MONGO_URI is set in your environment
MONGO_URI = os.environ.get('MONGO_URI')

if not MONGO_URI:
    print("WARNING: MONGO_URI not set. Database operations will fail.")
    client = None
    tokens_collection = None
else:
    try:
        # Create MongoClient with proper connection settings
        client = MongoClient(MONGO_URI)
        # Connect to database 'calendar_agent' and collection 'tokens'
        db = client.get_database('calendar_agent')
        tokens_collection = db.tokens
        print("Connected to MongoDB successfully.")
    except Exception as e:
        print(f"Failed to connect to MongoDB: {e}")
        client = None
        tokens_collection = None

# For this example, we use a fixed user ID since we aren't handling multi-user login sessions yet.
DEFAULT_USER_ID = "main_user"

# =============================================================================
# HELPER FUNCTIONS: DATABASE
# =============================================================================

def save_token_to_db(creds):
    """Saves the credentials object to MongoDB."""
    if tokens_collection is None:
        print("Error: Database not connected.")
        return False
    
    # Convert credentials to a standard Python dictionary
    creds_dict = json.loads(creds.to_json())
    
    # Update existing user or insert new one (Upsert)
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
        # Create Credentials object from the dictionary stored in DB
        return Credentials.from_authorized_user_info(user_doc["token_data"], SCOPES)
    
    return None

# =============================================================================
# GOOGLE CALENDAR SERVICE
# =============================================================================

class GoogleCalendarService:
    def __init__(self, credentials=None):
        self.creds = credentials
        self.service = None

        # Automatic Refresh Logic
        if self.creds:
            if not self.creds.valid:
                if self.creds.expired and self.creds.refresh_token:
                    try:
                        print("Token expired. Refreshing...")
                        self.creds.refresh(Request())
                        # SAVE THE REFRESHED TOKEN BACK TO MONGODB
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

# Global variable holder (service is instantiated per request context)
calendar_service = None

# =============================================================================
# LANGCHAIN TOOLS
# =============================================================================

@tool
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

@tool
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

@tool
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
# AGENT SETUP
# =============================================================================

def build_agent():
    tools = [get_current_time_and_events, schedule_event, resolve_conflict_reschedule]
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", AGENT_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False)

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
        
        # --- SAVE TO MONGODB ---
        success = save_token_to_db(credentials)
        if not success:
             return jsonify({"error": "Failed to save token to database"}), 500
        # -----------------------
        
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

        # Check for Gemini API key (support both GOOGLE_API_KEY and GEMINI_API_KEY)
        google_api_key = os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY')
        if not google_api_key:
            return jsonify({"error": "GOOGLE_API_KEY (or GEMINI_API_KEY) not set"}), 500
        
        # --- LOAD TOKEN FROM MONGODB ---
        creds = load_token_from_db()
        
        if not creds:
            return jsonify({
                "error": "User not authenticated. Please go to /api/auth first.", 
                "auth_required": True
            }), 401
            
        # Initialize Service with loaded creds
        # (This will trigger a refresh + DB update if the token is expired)
        calendar_service = GoogleCalendarService(credentials=creds)
        # -------------------------------
        
        if not calendar_service.service:
             return jsonify({"error": "Failed to initialize Calendar service with stored token"}), 401

        # Build agent
        agent_graph = build_agent()
        
        # Prepare messages
        messages = []
        for msg in conversation_history:
            if msg.get('role') == 'user':
                messages.append(HumanMessage(content=msg['content']))
            elif msg.get('role') == 'assistant':
                messages.append(AIMessage(content=msg['content']))
        
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        messages.append(HumanMessage(
            content=f"{user_message}\n\n(Current date: {current_date})"
        ))
        
        # Invoke agent
        result_state = agent_graph.invoke({
            "messages": messages,
            "agent_scratchpad": []
        })
        
        if isinstance(result_state, dict) and "output" in result_state:
            response_text = result_state["output"]
        else:
            return jsonify({"error": "Unexpected agent response format"}), 500
        
        return jsonify({
            "response": response_text,
            "success": True
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Render dynamically injects PORT, default to 5000 locally
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)