import os
import numpy as np
import cv2
from pathlib import Path

# Define image directory
img_dir = Path("src/api/static/img")
os.makedirs(img_dir, exist_ok=True)

# Define the size of all images
width, height = 800, 400

# Generate a bridge image with text
def create_bridge_image():
    # Create a blue-sky background with a brown bridge
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:height//2, :] = [240, 180, 100]  # Sky (BGR)
    img[height//2:, :] = [70, 130, 180]  # Water (BGR)
    
    # Draw a bridge
    cv2.rectangle(img, (100, height//2-50), (width-100, height//2), (30, 65, 155), -1)  # Bridge deck
    
    # Draw bridge supports
    for x in range(150, width-100, 100):
        cv2.rectangle(img, (x, height//2), (x+20, height), (30, 65, 155), -1)
    
    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "Bridge Infrastructure", (width//2-150, 50), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "Monitoring", (width//2-100, 90), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    
    return img

# Generate a dam image with text
def create_dam_image():
    # Create a background with water and concrete
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = [70, 130, 180]  # Water (BGR)
    
    # Draw a dam
    cv2.rectangle(img, (100, 50), (width-100, height), (150, 150, 150), -1)  # Dam (gray)
    
    # Draw dam features
    for y in range(100, height, 50):
        cv2.line(img, (100, y), (width-100, y), (100, 100, 100), 2)
    
    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "Dam Infrastructure", (width//2-150, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    return img

# Generate a building image with text
def create_building_image():
    # Create a background with sky
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = [225, 225, 225]  # Light sky (BGR)
    
    # Draw buildings
    for x in range(50, width-50, 100):
        h = np.random.randint(150, 300)
        w = np.random.randint(40, 80)
        cv2.rectangle(img, (x, height-h), (x+w, height), (100, 100, 100), -1)
        
        # Windows
        for wy in range(height-h+20, height-20, 30):
            for wx in range(x+10, x+w-10, 20):
                cv2.rectangle(img, (wx, wy), (wx+10, wy+15), (200, 230, 255), -1)
    
    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "Building Infrastructure", (width//2-180, 50), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    
    return img

# Generate a hero image
def create_hero_image():
    # Create a gradient background
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            img[y, x] = [int(255 * (y / height)), int(100 + 80 * (y / height)), int(200 - 150 * (y / height))]
    
    # Add some simple graphics
    cv2.circle(img, (width//4, height//2), 70, (255, 255, 255), 2)
    cv2.rectangle(img, (width//2, height//4), (width//2+100, height//4+100), (255, 255, 255), 2)
    cv2.line(img, (width//3*2, height//3), (width-50, height//3*2), (255, 255, 255), 3)
    
    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "AI-Driven Structural", (width//2-180, height//2-30), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "Health Monitoring", (width//2-160, height//2+30), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    
    return img

# Generate images and save them
images = {
    "bridges.jpg": create_bridge_image,
    "dams.jpg": create_dam_image,
    "buildings.jpg": create_building_image,
    "hero-image.jpg": create_hero_image,
}

for filename, create_func in images.items():
    img = create_func()
    cv2.imwrite(str(img_dir / filename), img)
    print(f"Created {filename}")

# Create additional images for the dashboard
dashboard_images = {
    "latest_inspection.jpg": create_bridge_image,
    "bridge_inspection.jpg": create_bridge_image,
}

for filename, create_func in dashboard_images.items():
    img = create_func()
    # Add a small difference to make it visually distinct
    cv2.rectangle(img, (10, 10), (width-10, height-10), (0, 0, 255), 3)
    cv2.putText(img, "Inspection Image", (width//2-120, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(img_dir / filename), img)
    print(f"Created {filename}")

print("\nAll images created successfully in src/api/static/img/") 