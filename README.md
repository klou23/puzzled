# TreeHacks 2026 - On-Call Help System

A hackathon project with a React + TypeScript frontend and Python FastAPI backend that creates instant Zoom meetings for help requests.

## 🚀 Features

- **Instant Help**: Click "Get Help" button to generate a Zoom meeting instantly
- **Zoom Integration**: Server-to-Server OAuth with automatic token management
- **Modern Stack**: React 19 + TypeScript + Vite frontend, FastAPI backend
- **Type-Safe**: Full TypeScript support with Pydantic models on the backend

## 📋 Prerequisites

- **Node.js** 18+ and npm
- **Python** 3.11+
- **Zoom Account** with Server-to-Server OAuth app ([Create one here](https://marketplace.zoom.us/develop/create))

## 🛠️ Project Structure

```
TreeHacks26/
├── src/                      # React frontend
│   ├── components/          # React components
│   │   └── HelpButton.tsx   # Help request button
│   ├── services/            # API client
│   │   └── api.ts
│   ├── types/               # TypeScript types
│   │   └── api.ts
│   └── App.tsx              # Main app component
├── backend/                  # Python FastAPI backend
│   └── app/
│       ├── main.py          # FastAPI entry point
│       ├── config.py        # Environment configuration
│       ├── routes/          # API endpoints
│       │   └── help.py      # /api/help endpoint
│       ├── services/        # Business logic
│       │   └── zoom_client.py  # Zoom API integration
│       └── models/          # Data models
│           └── schemas.py   # Pydantic schemas
└── README.md
```

## 🎯 Quick Start

### 1. Frontend Setup

```bash
# Install dependencies
npm install

# Create environment file (optional for local dev)
cp .env.example .env

# Start development server
npm run dev
```

Frontend runs on: http://localhost:5173

### 2. Backend Setup

Quick setup (for full details, see [backend/README.md](backend/README.md)):

```bash
cd backend

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux (or venv\Scripts\activate on Windows)

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your Zoom credentials (get them from https://marketplace.zoom.us/develop/create)

# Run server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend runs on: http://localhost:8000

📚 **For detailed backend documentation, troubleshooting, and development guides, see [backend/README.md](backend/README.md)**

### 3. Test the Integration

1. Open http://localhost:5173 in your browser
2. Click the **"Get Help"** button
3. A new tab should open with your Zoom meeting URL!

## 🧑‍💻 Development Workflow

### Running Both Servers

You'll need **two terminal windows**:

**Terminal 1 - Backend:**
```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**
```bash
# From project root
npm run dev
```

### Adding Backend Features

See [backend/README.md](backend/README.md) for detailed instructions on:
- Setting up your development environment
- Adding new API endpoints
- Code structure and best practices
- Testing your changes

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check for monitoring |
| `/docs` | GET | Interactive API documentation |
| `/api/help` | POST | Create Zoom help meeting |

### Example: Create Meeting

```bash
curl -X POST http://localhost:8000/api/help
```

Response:
```json
{
  "joinUrl": "https://zoom.us/j/123456789",
  "startUrl": "https://zoom.us/s/123456789?zak=...",
  "meetingId": "123456789"
}
```

## 🚢 Deployment

### Backend (Render)

- **Build Command:** `cd backend && pip install -r requirements.txt`
- **Start Command:** `cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- **Environment Variables:** Add Zoom credentials and `CORS_ORIGINS`

For detailed deployment instructions and environment variable configuration, see [backend/README.md](backend/README.md)

### Frontend (Render Static Site or Vercel)

- **Build Command:** `npm run build`
- **Publish Directory:** `dist`
- **Environment Variable:** `VITE_API_BASE_URL=https://your-backend.onrender.com`

## 🔧 Troubleshooting

### Common Issues

**CORS Errors:** Check that `CORS_ORIGINS` in `backend/.env` includes your frontend URL (`http://localhost:5173`)

**Port Already in Use:** If port 8000 is taken, kill the process:
```bash
lsof -ti:8000 | xargs kill -9  # macOS/Linux
```

**Frontend can't connect:** Make sure both servers are running in separate terminals

For detailed backend troubleshooting (virtual environment, Zoom API errors, etc.), see [backend/README.md](backend/README.md)

## 🤝 Contributing

1. Test your changes locally before pushing
2. For backend work, see [backend/README.md](backend/README.md) for development guidelines
3. For frontend work, ensure TypeScript types are properly defined
4. Use the interactive API docs at `/docs` for backend testing

## 📝 Tech Stack

**Frontend:**
- React 19
- TypeScript
- Vite
- Native fetch API

**Backend:**
- Python 3.11+
- FastAPI
- Pydantic
- httpx (async HTTP client)
- uvicorn (ASGI server)

**Infrastructure:**
- Zoom Meeting API (Server-to-Server OAuth)
- Render (deployment)


