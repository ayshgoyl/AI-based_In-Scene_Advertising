import cv2
import numpy as np

# Generate a synthetic video with a moving textured rectangle and an occluder
fps = 30
width, height = 640, 480
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('test_video.mp4', fourcc, fps, (width, height))

for i in range(90):  # 3 seconds
    frame = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Moving rectangle (surface)
    rect_x = 100 + int(i * 1.5)
    rect_y = 100
    rect_w = 400
    rect_h = 250
    
    # Draw bounding box for the surface
    cv2.rectangle(frame, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (200, 200, 200), -1)
    
    # Draw some "features" inside it
    import random
    random.seed(42) 
    for _ in range(50):
        # We need to compute features relative to the rectangle's top left
        fx = random.randint(10, rect_w - 10)
        fy = random.randint(10, rect_h - 10)
        cv2.circle(frame, (rect_x + fx, rect_y + fy), 5, (50, 50, 50), -1)
        
    # Moving occluder
    occ_x = 50 + int(i * 6)
    occ_y = 225
    cv2.circle(frame, (occ_x, occ_y), 40, (255, 0, 0), -1)
    
    out.write(frame)
out.release()
print("Generated test_video.mp4")

# Generate a logo
logo = np.zeros((100, 200, 4), dtype=np.uint8)
logo[:, :] = (0, 255, 0, 255) # Green
cv2.putText(logo, 'AD', (70, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0, 255), 3)
cv2.imwrite('test_logo.png', logo)
print("Generated test_logo.png")
