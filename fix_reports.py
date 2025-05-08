"""
Fix Chart.js initialization issues in reports.html template
"""
import re
import os
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).parent.absolute()
reports_template = project_root / "src" / "api" / "templates" / "reports.html"

print(f"Attempting to fix Chart.js initialization in {reports_template}")

try:
    # Read the template file
    with open(reports_template, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create a backup of the original file
    backup_file = reports_template.with_suffix('.html.bak')
    with open(backup_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Created backup at {backup_file}")
    
    # Fix the chart initialization functions
    # Look for initTrendsChart, initPieChart, and initBarChart functions
    
    # Pattern for functions that initialize charts
    init_pattern = r'function init(\w+)Chart\(\) \{(.*?)const ctx = document\.getElementById\([\'"](\w+)Chart[\'"]\)\.getContext\([\'"]2d[\'"]\);(.*?)new Chart\(ctx,'
    
    # Replacement for the pattern - add null check for chart element
    replacement = r'function init\1Chart() {\2const chartElement = document.getElementById("\3Chart");\n    if (!chartElement) return;\n    const ctx = chartElement.getContext("2d");\3new Chart(ctx,'
    
    # Apply the fix using regex with re.DOTALL to match across multiple lines
    modified_content = re.sub(init_pattern, replacement, content, flags=re.DOTALL)
    
    # Write the modified content back
    with open(reports_template, 'w', encoding='utf-8') as f:
        f.write(modified_content)
    
    print(f"Successfully updated Chart.js initialization in {reports_template}")
    
    # More general fix for chart initialization
    # Also fix initialization in the DOMContentLoaded event handler
    print("Looking for DOMContentLoaded event handler to fix chart initialization...")
    
    dom_content_loaded_pattern = r'document\.addEventListener\([\'"]DOMContentLoaded[\'"]\s*,\s*function\(\) \{(.*?)initTrendsChart\(\);(.*?)initPieChart\(\);(.*?)initBarChart\(\);'
    dom_content_loaded_replacement = r'document.addEventListener("DOMContentLoaded", function() {\1try {\n        initTrendsChart();\2initPieChart();\3initBarChart();\n    } catch (e) {\n        console.error("Error initializing charts:", e);\n    }'
    
    # Apply the fix to the DOMContentLoaded handler
    final_content = re.sub(dom_content_loaded_pattern, dom_content_loaded_replacement, modified_content, flags=re.DOTALL)
    
    # Write the final content back
    with open(reports_template, 'w', encoding='utf-8') as f:
        f.write(final_content)
    
    print(f"Successfully updated DOMContentLoaded handler in {reports_template}")
    print("Chart.js initialization fixes complete")
    
except Exception as e:
    print(f"Error fixing Chart.js initialization: {e}") 