"""
Script to diagnose issues with the reports page and check if the template exists.
"""
import os
import sys
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).parent.absolute()

# Check if the templates directory exists
templates_dir = project_root / "src" / "api" / "templates"
if not os.path.exists(templates_dir):
    print(f"Error: Templates directory not found at {templates_dir}")
    sys.exit(1)

print(f"Templates directory found at {templates_dir}")

# Check if the reports.html template exists
reports_template = templates_dir / "reports.html"
if not os.path.exists(reports_template):
    print(f"Error: Reports template not found at {reports_template}")
    sys.exit(1)

print(f"Reports template found at {reports_template}")

# Check if the dashboard.html template exists
dashboard_template = templates_dir / "dashboard.html"
if not os.path.exists(dashboard_template):
    print(f"Error: Dashboard template not found at {dashboard_template}")
    sys.exit(1)

print(f"Dashboard template found at {dashboard_template}")

# Check file permissions
try:
    with open(reports_template, 'r') as f:
        first_line = f.readline()
        print(f"Reports template is readable, first line: {first_line.strip()}")
except Exception as e:
    print(f"Error reading reports template: {e}")
    
# Check app.py route for reports
app_py = project_root / "src" / "api" / "app.py"
if not os.path.exists(app_py):
    print(f"Error: app.py not found at {app_py}")
    sys.exit(1)

print(f"app.py found at {app_py}")

# Print the python path
print(f"Python path: {sys.path}")

# Check template loading mechanism in app.py
try:
    with open(app_py, 'r') as f:
        app_content = f.read()
        templates_import = "templates = Jinja2Templates"
        if templates_import in app_content:
            print(f"Found templates initialization in app.py")
        else:
            print(f"Warning: Could not find templates initialization in app.py")
            
        reports_route = "@app.get(\"/reports\""
        if reports_route in app_content:
            print(f"Found reports route in app.py")
        else:
            print(f"Warning: Could not find reports route in app.py")
except Exception as e:
    print(f"Error reading app.py: {e}")

print("Diagnosis complete. Check the error logs for more information.") 