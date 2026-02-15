# TreeHacks 2026 - Backend API

FastAPI backend for the On-Call Help System with Zoom Meeting integration.

## 📋 Prerequisites

- **Python** 3.11+
- **Zoom Account** with Server-to-Server OAuth app ([Create one here](https://marketplace.zoom.us/develop/create))

## 🛠️ Backend Structure

```
backend/
├── app/
│   ├── main.py          # FastAPI entry point
│   ├── config.py        # Environment configuration
│   ├── routes/          # API endpoints
│   │   └── help.py      # /api/help endpoint
│   ├── services/        # Business logic
│   │   └── zoom_client.py  # Zoom API integration
│   └── models/          # Data models
│       └── schemas.py   # Pydantic schemas
├── requirements.txt     # Python dependencies
└── .env.example         # Environment template
```

## 🚀 Setup Instructions

### Step 1: Create Virtual Environment

```bash
cd backend

# Create virtual environment
python3 -m venv venv

# Activate virtual environment

# macOS/Linux
source venv/bin/activate
# OR windows
venv\Scripts\activate
```

**💡 Tip**: You'll see `(venv)` in your terminal when activated.

### Step 2: Install Dependencies

```bash
# Make sure venv is activated!
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables

```bash
# Create .env file from template
cp .env.example .env

# Edit .env with your Zoom credentials
nano .env  # or use your preferred editor
```

**Required variables in `.env`:**
```env
ZOOM_ACCOUNT_ID=your_account_id_here
ZOOM_CLIENT_ID=your_client_id_here
ZOOM_CLIENT_SECRET=your_client_secret_here
CORS_ORIGINS=http://localhost:5173
ENVIRONMENT=development
LOG_LEVEL=INFO
```

**🔑 Getting Zoom Credentials:**
1. Go to [Zoom Marketplace](https://marketplace.zoom.us/develop/create)
2. Create a "Server-to-Server OAuth" app
3. Add the following scopes:
   - `meeting:write` (required)
   - `meeting:read` (optional)
4. Copy Account ID, Client ID, and Client Secret to your `.env` file

### Step 4: Run Backend Server

```bash
# Make sure you're in backend/ directory with venv activated
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend runs on: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check for monitoring |
| `/docs` | GET | Interactive API documentation (Swagger UI) |
| `/api/help` | POST | Create Zoom help meeting |

### Example: Create Meeting

**Request:**
```bash
curl -X POST http://localhost:8000/api/help
```

**Response:**
```json
{
  "joinUrl": "https://zoom.us/j/123456789",
  "startUrl": "https://zoom.us/s/123456789?zak=...",
  "meetingId": "123456789"
}
```

**Error Response:**
```json
{
  "detail": "Failed to create Zoom meeting: ..."
}
```

## 🧑‍💻 Development Workflow

### Adding New Features

1. **Activate virtual environment first:**
   ```bash
   cd backend
   source venv/bin/activate
   ```

2. **Install new packages:**
   ```bash
   pip install package-name
   pip freeze > requirements.txt  # Update requirements
   ```

3. **Code structure guidelines:**
   - **Routes** (`app/routes/`): Define API endpoints
   - **Services** (`app/services/`): Business logic and external API integrations
   - **Models** (`app/models/schemas.py`): Pydantic request/response schemas
   - **Config** (`app/config.py`): Environment configuration

4. **Adding a new endpoint:**
   ```python
   # app/routes/new_feature.py
   from fastapi import APIRouter

   router = APIRouter()

   @router.post("/api/new-feature")
   async def new_feature():
       return {"message": "Hello!"}
   ```

   ```python
   # app/main.py
   from app.routes import help, new_feature

   app.include_router(help.router)
   app.include_router(new_feature.router)  # Add this
   ```

5. **Test your changes:**
   - Visit interactive docs: http://localhost:8000/docs
   - Or use curl/httpie for testing

### Running Tests (Future)

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest
```

### Deactivating Virtual Environment

When you're done working:
```bash
deactivate
```

## 🔧 Troubleshooting

### Virtual Environment Issues

**Problem:** `pip: command not found` after activating venv

**Solution:** Make sure you see `(venv)` in your prompt. Try deactivating and reactivating:
```bash
deactivate
source venv/bin/activate
```

**Problem:** `ModuleNotFoundError` when running uvicorn

**Solution:** Ensure venv is activated and dependencies are installed:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Zoom API Errors

**Problem:** "Failed to create Zoom meeting"

**Solution:**
1. Verify credentials in `.env` file are correct
2. Check that your Zoom app has the `meeting:write` scope enabled
3. Verify the Zoom app is activated (not in draft mode)
4. Check backend logs for detailed error messages:
   ```bash
   # Look for lines starting with ERROR
   uvicorn app.main:app --reload --log-level debug
   ```

**Problem:** 401 Unauthorized from Zoom API

**Solution:** Your OAuth credentials are invalid. Double-check:
- `ZOOM_ACCOUNT_ID` matches your account
- `ZOOM_CLIENT_ID` and `ZOOM_CLIENT_SECRET` are from a Server-to-Server OAuth app
- The app is activated in Zoom Marketplace

### CORS Errors

**Problem:** Frontend can't reach backend (CORS error in browser console)

**Solution:** Update `CORS_ORIGINS` in `.env`:
```env
# For local development
CORS_ORIGINS=http://localhost:5173,http://localhost:5174

# For production
CORS_ORIGINS=https://your-frontend-domain.com
```

### Port Already in Use

**Problem:** `Address already in use: 8000`

**Solution:** Kill the process or use a different port:
```bash
# macOS/Linux - kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Windows - kill process on port 8000
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Or use a different port
uvicorn app.main:app --reload --port 8001
```

## 🚢 Deployment (Render)

### Configuration

1. Create a new **Web Service** on [Render](https://render.com)
2. Connect your GitHub repository
3. Configure build settings:
   - **Root Directory**: Leave blank or set to `/`
   - **Build Command**: `cd backend && pip install -r requirements.txt`
   - **Start Command**: `cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT`

### Environment Variables

Add these in the Render dashboard (keep them secret):

| Variable | Value | Notes |
|----------|-------|-------|
| `ZOOM_ACCOUNT_ID` | `your_account_id` | From Zoom OAuth app |
| `ZOOM_CLIENT_ID` | `your_client_id` | From Zoom OAuth app |
| `ZOOM_CLIENT_SECRET` | `your_client_secret` | From Zoom OAuth app |
| `CORS_ORIGINS` | `https://your-frontend.com` | Your frontend URL |
| `ENVIRONMENT` | `production` | |
| `LOG_LEVEL` | `INFO` | Optional |

### Health Checks

Render will automatically monitor the `/health` endpoint. If it returns 200 OK, the service is considered healthy.

### Logs

View logs in the Render dashboard to debug issues:
- Look for `INFO` logs for successful operations
- Look for `ERROR` logs for failures

## 📝 Tech Stack

- **FastAPI** - Modern Python web framework
- **Pydantic** - Data validation and settings management
- **httpx** - Async HTTP client for Zoom API calls
- **uvicorn** - ASGI server for running FastAPI
- **python-dotenv** - Environment variable loading

## 🔐 Security Notes

1. **Never commit `.env` files** - They're in `.gitignore` for a reason
2. **Rotate credentials** if accidentally exposed
3. **Use environment variables** for all secrets
4. **CORS configuration**: Only whitelist necessary origins
5. **Rate limiting**: Consider adding rate limiting for production (not implemented yet)

## 🤝 Contributing

1. Always activate the virtual environment before working
2. Keep `requirements.txt` updated when adding packages
3. Follow the existing code structure (routes/services/models)
4. Test your changes with the interactive docs at `/docs`
5. Add error handling for external API calls
6. Log important events (use `logger.info()` and `logger.error()`)

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Zoom Server-to-Server OAuth Guide](https://developers.zoom.us/docs/internal-apps/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Render Deployment Guide](https://render.com/docs/deploy-fastapi)
