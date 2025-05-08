"""
Fix template pages by adding defensive JavaScript coding practices
"""
import os
import re
import glob
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).parent.absolute()
templates_dir = project_root / "src" / "api" / "templates"

def fix_javascript_init(file_path):
    """Fix JavaScript initialization in HTML templates"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Create a backup of the original file
        backup_file = Path(str(file_path) + '.bak')
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Created backup at {backup_file}")
        
        # Add defensive check for DOM elements before using them
        # Find document.getElementById calls directly followed by method calls
        element_access_pattern = r'document\.getElementById\([\'"](\w+)[\'"]\)(\.[\w\.]+\([^)]*\))'
        element_access_replacement = r'const el_\1 = document.getElementById("\1"); if (el_\1) el_\1\2'
        
        # Apply the fix
        modified_content = re.sub(element_access_pattern, element_access_replacement, content)
        
        # Wrap DOMContentLoaded event handler with try/catch
        dom_loaded_pattern = r'(document\.addEventListener\([\'"]DOMContentLoaded[\'"]\s*,\s*function\(\) \{)(.*?)(\}\);)'
        dom_loaded_replacement = r'\1 try {\2} catch (error) { console.error("Error in DOMContentLoaded:", error); }\3'
        
        # Apply the fix
        modified_content = re.sub(dom_loaded_pattern, dom_loaded_replacement, modified_content, flags=re.DOTALL)
        
        # Fix chart initialization if it exists
        if 'Chart(' in modified_content:
            # Add defensive check for chart elements
            chart_init_pattern = r'(const \w+Ctx = document\.getElementById\([\'"](\w+)[\'"]\))(\.getContext\([\'"]2d[\'"]\))'
            chart_init_replacement = r'\1; if (!\1) return; \1\3'
            
            # Apply the fix
            modified_content = re.sub(chart_init_pattern, chart_init_replacement, modified_content)
        
        # Write the modified content back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        
        return True
    except Exception as e:
        print(f"Error fixing JavaScript in {file_path}: {e}")
        return False

# Process all HTML templates
html_files = glob.glob(str(templates_dir / "*.html"))
successful_fixes = 0

print(f"Found {len(html_files)} HTML templates to process")

for html_file in html_files:
    file_path = Path(html_file)
    print(f"Processing {file_path.name}...")
    if fix_javascript_init(file_path):
        successful_fixes += 1

print(f"Successfully fixed {successful_fixes} out of {len(html_files)} templates")

# Specifically fix dashboard.js for negative values
dashboard_js = project_root / "src" / "api" / "static" / "js" / "dashboard.js"
if os.path.exists(dashboard_js):
    print(f"Fixing dashboard.js for negative values...")
    try:
        with open(dashboard_js, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Add fix for negative values if not already present
        if "fixDashboardStats" not in content:
            # Find the dashboard initialization function
            dom_loaded_pattern = r'(document\.addEventListener\([\'"]DOMContentLoaded[\'"]\s*,\s*function\(\) \{)(.*?)(\}\);)'
            
            # Add the fixDashboardStats call
            fix_call = """
    // Fix dashboard statistics if they are negative
    const statElements = document.querySelectorAll('.card .display-4, .card h1');
    statElements.forEach(element => {
        const value = element.textContent.trim();
        if (value.startsWith('-')) {
            // Replace negative value with positive one
            const positiveValue = value.substring(1);
            element.textContent = positiveValue;
        }
    });
"""
            # Add our fix after the initialization
            dom_loaded_replacement = r'\1\2' + fix_call + r'\3'
            modified_content = re.sub(dom_loaded_pattern, dom_loaded_replacement, content, flags=re.DOTALL)
            
            # Write the modified content back
            with open(dashboard_js, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            
            print("Successfully fixed dashboard.js")
        else:
            print("dashboard.js already contains the fix")
    except Exception as e:
        print(f"Error fixing dashboard.js: {e}")
else:
    print(f"dashboard.js not found at {dashboard_js}")

print("All fixes completed") 