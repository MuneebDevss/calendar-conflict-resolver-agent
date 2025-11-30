# Vercel Deployment Guide

## Prerequisites
1. A Vercel account
2. Vercel CLI installed: `npm install -g vercel`
3. Google Cloud Console OAuth credentials
4. OpenAI API key

## Setup Steps

### 1. Prepare Google OAuth Credentials
1. Go to Google Cloud Console → APIs & Services → Credentials
2. Create OAuth 2.0 Client ID (Web application type)
3. Add authorized redirect URIs:
   - `https://your-vercel-app.vercel.app/callback`
   - `http://localhost:3000/callback` (for testing)
4. Download the credentials JSON

### 2. Set Environment Variables in Vercel

Go to your Vercel project settings → Environment Variables and add:

```bash
OPENAI_API_KEY=your_openai_api_key
GOOGLE_CREDENTIALS={"web":{"client_id":"...","client_secret":"...","redirect_uris":["..."]}}
REDIRECT_URI=https://your-vercel-app.vercel.app/callback
```

**Important:** The `GOOGLE_CREDENTIALS` should be the entire JSON content from your Google credentials file (as a single line string).

### 3. Deploy to Vercel

```bash
# Login to Vercel
vercel login

# Deploy
vercel --prod
```

## API Endpoints

### 1. Get OAuth URL
```bash
GET /api/auth
```

Response:
```json
{
  "auth_url": "https://accounts.google.com/o/oauth2/auth?...",
  "state": "random_state_string"
}
```

### 2. Exchange Code for Token
```bash
POST /api/callback
Content-Type: application/json

{
  "code": "authorization_code_from_google"
}
```

Response:
```json
{
  "token": "{\"token\": \"...\", \"refresh_token\": \"...\"}",
  "message": "Authentication successful"
}
```

### 3. Chat with Agent
```bash
POST /api/chat
Content-Type: application/json

{
  "message": "List my events for this week",
  "token": "{\"token\": \"...\", \"refresh_token\": \"...\"}",
  "history": [
    {"role": "user", "content": "previous message"},
    {"role": "assistant", "content": "previous response"}
  ]
}
```

Response:
```json
{
  "response": "Here are your events for this week...",
  "success": true
}
```

## Frontend Integration Example

```javascript
// 1. Get auth URL
const authResponse = await fetch('https://your-app.vercel.app/api/auth');
const { auth_url } = await authResponse.json();

// 2. Redirect user to auth_url
window.location.href = auth_url;

// 3. In your callback page, exchange code for token
const code = new URLSearchParams(window.location.search).get('code');
const tokenResponse = await fetch('https://your-app.vercel.app/api/callback', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ code })
});
const { token } = await tokenResponse.json();

// 4. Store token (localStorage, session, etc.)
localStorage.setItem('calendar_token', token);

// 5. Chat with agent
const chatResponse = await fetch('https://your-app.vercel.app/api/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    message: 'Show me my calendar for next week',
    token: localStorage.getItem('calendar_token'),
    history: []
  })
});
const { response } = await chatResponse.json();
console.log(response);
```

## Testing Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your_key"
export GOOGLE_CREDENTIALS='{"web":{...}}'
export REDIRECT_URI="http://localhost:3000/callback"

# Run Flask app
python api/index.py
```

## Notes

- The original CLI version is preserved in `calendar_agent_cli.py`
- The Vercel deployment uses a stateless API approach
- Authentication tokens must be passed with each request
- Consider implementing token refresh logic on the client side
- For production, use secure token storage (not localStorage)
