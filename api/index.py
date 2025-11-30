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

# LangChain Imports
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# =============================================================================
# CONFIGURATION
# =============================================================================

SCOPES = ['https://www.googleapis.com/auth/calendar']
MODEL_NAME = "gpt-4o"
AGENT_SYSTEM_PROMPT = """You are an expert Executive Assistant and Calendar Conflict Resolver.\n\nResponsibilities:\n1. Manage the user's Google Calendar.\n2. Detect conflicts (overlapping events).\n3. Propose and execute solutions (rescheduling).\n\nGuidelines:\n- ALWAYS call `get_current_time_and_events` before proposing time slots or editing events.\n- When resolving a conflict, look for the first available empty slot that fits the duration.\n- If the user references a meeting by name, list events to find the corresponding ID before acting.\n- Communicate clearly, be polite, and keep responses action-focused.\n- The user may append the current date in their request; rely on it for temporal context."""

app = Flask(__name__)

# =============================================================================
# GOOGLE CALENDAR SERVICE
# =============================================================================

class GoogleCalendarService:
    def __init__(self, credentials_dict: dict = None, token_dict: dict = None):
        self.creds = None
        if token_dict:
            self.creds = Credentials.from_authorized_user_info(token_dict, SCOPES)
        elif credentials_dict:
            # For service initialization without token
            pass
        else:
            raise ValueError("Either credentials or token must be provided")
        
        if self.creds and self.creds.valid:
            self.service = build('calendar', 'v3', credentials=self.creds)
        else:
            self.service = None

    def refresh_credentials(self, credentials_dict: dict):
        """Refresh expired credentials"""
        if self.creds and self.creds.expired and self.creds.refresh_token:
            self.creds.refresh(Request())
            self.service = build('calendar', 'v3', credentials=self.creds)
            return self.creds.to_json()
        return None

    def list_events(self, days: int = 7) -> str:
        """List events for the next X days."""
        if not self.service:
            return "Calendar service not initialized"
        
        now = datetime.datetime.utcnow().isoformat() + 'Z'
        end_date = (datetime.datetime.utcnow() + datetime.timedelta(days=days)).isoformat() + 'Z'
        
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
        event = self.service.events().insert(calendarId='primary', body=event).execute()
        return f"Event created: {event.get('htmlLink')}"

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

    def delete_event(self, event_id: str):
        if not self.service:
            return "Calendar service not initialized"
        
        try:
            self.service.events().delete(calendarId='primary', eventId=event_id).execute()
            return f"Event {event_id} deleted successfully."
        except Exception as e:
            return f"Error deleting event: {str(e)}"

# Global variable to hold service (will be initialized per request)
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
    if not calendar_service:
        return "Calendar service not initialized"
    
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
    Reschedule an existing event (identified by ID) to a new start time to resolve a conflict.
    The duration of the event will be preserved.
    new_start_time must be in ISO format.
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
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", AGENT_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False)

def _message_content_to_text(message: AIMessage) -> str:
    """Normalize LangChain message content into printable text."""
    if isinstance(message.content, str):
        return message.content

    if isinstance(message.content, list):
        text_chunks = []
        for block in message.content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_chunks.append(block.get("text", ""))
            else:
                text_chunks.append(str(block))
        return "\n".join(chunk for chunk in text_chunks if chunk)

    return str(message.content)

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Google Calendar Conflict Resolver API",
        "endpoints": {
            "/api/auth": "GET - Get OAuth URL",
            "/api/callback": "POST - Handle OAuth callback",
            "/api/chat": "POST - Chat with agent"
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
    """Handle OAuth callback and exchange code for token"""
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
        
        return jsonify({
            "token": credentials.to_json(),
            "message": "Authentication successful"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Process user message with the agent"""
    global calendar_service
    
    try:
        data = request.json
        user_message = data.get('message')
        token = data.get('token')
        conversation_history = data.get('history', [])
        
        if not user_message:
            return jsonify({"error": "Message required"}), 400
        
        if not token:
            return jsonify({"error": "Authentication token required"}), 401
        
        # Check for OpenAI API key
        if not os.environ.get('OPENAI_API_KEY'):
            return jsonify({"error": "OPENAI_API_KEY not set"}), 500
        
        # Initialize calendar service with token
        token_dict = json.loads(token) if isinstance(token, str) else token
        calendar_service = GoogleCalendarService(token_dict=token_dict)
        
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
        
        # AgentExecutor returns output directly
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

# For Vercel serverless deployment
if __name__ != "__main__":
    # Vercel will use the Flask app directly
    app = app
