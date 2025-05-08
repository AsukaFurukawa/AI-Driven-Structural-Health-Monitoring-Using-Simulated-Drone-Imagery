"""
Simplified version of app.py to fix the reports route and dashboard statistics.
"""
import os
import sys
import logging
import yaml
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("api")

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
STATIC_IMG_DIR = BASE_DIR / "src" / "api" / "static" / "img"
os.makedirs(STATIC_IMG_DIR, exist_ok=True)

# Templates
templates_dir = BASE_DIR / "src" / "api" / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# Static files
static_dir = BASE_DIR / "src" / "api" / "static"

# Initialize FastAPI app
app = FastAPI(
    title="Structural Health Monitoring API",
    description="API for analyzing infrastructure defects using AI and drone imagery",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Sample data for dashboard
SAMPLE_DATA = {
    "total_structures": 25,
    "healthy_structures": 18,
    "healthy_percentage": 72,
    "warning_structures": 5, 
    "critical_structures": 2,
    "latest_inspection": {
        "id": "insp-20250312-001",
        "structure_name": "Bridge A-123",
        "date": "March 12, 2025",
        "status": "Warning",
        "status_color": "warning",
        "defects_count": 3
    },
    "alerts": [
        {
            "id": "alt-001",
            "title": "Critical crack detected",
            "description": "A 25cm crack was detected on the main support beam",
            "severity": "Critical",
            "severity_color": "danger",
            "structure_name": "Bridge A-123",
            "time_ago": "2 hours ago"
        },
        {
            "id": "alt-002",
            "title": "Corrosion detected",
            "description": "Early signs of corrosion detected on steel reinforcement",
            "severity": "Warning",
            "severity_color": "warning",
            "structure_name": "Highway Overpass B-456",
            "time_ago": "Yesterday"
        },
        {
            "id": "alt-003",
            "title": "Inspection scheduled",
            "description": "Routine inspection scheduled for tomorrow",
            "severity": "Info",
            "severity_color": "info",
            "structure_name": "Railway Bridge C-789",
            "time_ago": "3 days ago"
        }
    ],
    "recent_inspections": [
        {
            "structure_name": "Bridge A-123",
            "date": "March 12, 2025",
            "description": "Routine inspection detected minor cracks",
            "status": "Warning",
            "status_color": "warning"
        },
        {
            "structure_name": "Tunnel E-202",
            "date": "March 10, 2025",
            "description": "Detailed inspection completed",
            "status": "Healthy",
            "status_color": "success"
        },
        {
            "structure_name": "Building D-101",
            "date": "March 5, 2025",
            "description": "Structural integrity verified",
            "status": "Healthy",
            "status_color": "success"
        }
    ],
    "structures": [
        {"id": 1, "name": "Howrah Bridge"},
        {"id": 2, "name": "Bandra-Worli Sea Link"},
        {"id": 3, "name": "Bhakra Dam"},
        {"id": 4, "name": "Delhi Metro Bridge"},
        {"id": 5, "name": "Taj Mahal"}
    ]
}

def get_statistics():
    """Get monitoring statistics."""
    return {
        "total_structures": SAMPLE_DATA["total_structures"],
        "healthy_structures": SAMPLE_DATA["healthy_structures"],
        "warning_structures": SAMPLE_DATA["warning_structures"],
        "critical_structures": SAMPLE_DATA["critical_structures"]
    }

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the home page."""
    # Get data from services for consistent UI
    statistics = get_statistics()
    
    # Build template context
    data = {
        "request": request,
        "current_date": datetime.now().strftime("%B %d, %Y"),
        **statistics,
        **SAMPLE_DATA
    }
    return templates.TemplateResponse("home.html", data)

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Render the dashboard page."""
    # In a real application, this data would come from a database or other source
    data = {
        "request": request,
        "current_date": datetime.now().strftime("%B %d, %Y"),
        **SAMPLE_DATA
    }
    return templates.TemplateResponse("dashboard.html", data)

@app.get("/reports", response_class=HTMLResponse)
async def reports(request: Request):
    """Render the reports page."""
    try:
        # Get data from services for consistent UI
        statistics = get_statistics()
        
        # Build template context
        data = {
            "request": request,
            "current_date": datetime.now().strftime("%B %d, %Y"),
            **statistics,
            **SAMPLE_DATA,
            "page_title": "Reports & Analytics",
            "active_page": "reports"
        }
        
        logger.info(f"Rendering reports page with data: {data.keys()}")
        response = templates.TemplateResponse("reports.html", data)
        return response
    except Exception as e:
        logger.error(f"Error rendering reports page: {str(e)}")
        return HTMLResponse(content=f"""
        <html>
            <head><title>Error</title></head>
            <body>
                <h1>Error rendering reports page</h1>
                <p>{str(e)}</p>
                <p>Check server logs for more details.</p>
            </body>
        </html>
        """)

@app.get("/settings", response_class=HTMLResponse)
async def settings(request: Request):
    """Render the settings page."""
    # Get data from services for consistent UI
    statistics = get_statistics()
    
    # Build template context
    data = {
        "request": request,
        "current_date": datetime.now().strftime("%B %d, %Y"),
        **statistics,
        "page_title": "System Settings",
        "active_page": "settings"
    }
    
    return templates.TemplateResponse("settings.html", data)

@app.get("/analysis", response_class=HTMLResponse)
async def analysis(request: Request):
    """Render the analysis page."""
    # Get data from services for consistent UI
    statistics = get_statistics()
    
    # Build template context
    data = {
        "request": request,
        "current_date": datetime.now().strftime("%B %d, %Y"),
        **statistics,
        "page_title": "Structural Analysis",
        "active_page": "analysis"
    }
    
    return templates.TemplateResponse("analysis.html", data)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 