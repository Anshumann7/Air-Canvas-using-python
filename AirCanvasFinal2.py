import cv2 
import numpy as np 
import mediapipe as mp 
import math

# Initialize Mediapipe Hand Tracking with high accuracy 
mp_hands = mp.solutions.hands 
hands = mp_hands.Hands(min_detection_confidence=0.95, min_tracking_confidence=0.95) 
mp_draw = mp.solutions.drawing_utils 
mp_styles = mp.solutions.drawing_styles 

# Initialize Canvas and Parameters 
canvas = None 
prev_x, prev_y = None, None 
color = (255, 0, 0)  # Default color: Blue 
brush_size = 8  # Larger brush for smoother drawing 
erase_mode = False  # Erase mode toggle

# Color palette
color_palette = { 
    "blue": (255, 0, 0), 
    "green": (0, 255, 0), 
    "red": (0, 0, 255), 
    "yellow": (0, 255, 255),
    "purple": (255, 0, 255),
    "white": (255, 255, 255)
} 

# Track button interaction state
prev_selection = None
pinch_reset_active = False  # Track reset pinch gesture
drawing_pinch_active = False  # Track draw pinch gesture

def get_color_selection(x, y): 
    if y < 70:  # Increased from 60 to 70 to accommodate larger buttons
        # Color boxes
        if 50 < x < 100: 
            return "blue" 
        elif 110 < x < 160: 
            return "green" 
        elif 170 < x < 220: 
            return "red" 
        elif 230 < x < 280: 
            return "yellow"
        elif 290 < x < 340: 
            return "purple"
        elif 350 < x < 400: 
            return "white"
        # Reset and Erase buttons (on the side)
        elif 450 < x < 530 and 10 < y < 60:  # Box area only
            return "reset"
        elif 550 < x < 630 and 10 < y < 60:  # Box area only
            return "erase"
    return None 

# Start Video Capture 
cap = cv2.VideoCapture(0) 

# Create windows
cv2.namedWindow("Air Canvas", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Air Canvas", 800, 600)
cv2.namedWindow("Canvas", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Canvas", 800, 600)
 
while cap.isOpened(): 
    ret, frame = cap.read() 
    if not ret: 
        break 
 
    frame = cv2.flip(frame, 1) 
    h, w, c = frame.shape 
 
    if canvas is None: 
        canvas = np.zeros((h, w, 3), dtype=np.uint8) 
 
    # Detect Hands 
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    results = hands.process(rgb_frame) 

    current_selection = None
    drawing_hand_detected = False
    pinch_reset_detected = False
    pinch_draw_detected = False

    if results.multi_hand_landmarks: 
        # Draw all hands
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks( 
                frame,  
                hand_landmarks,  
                mp_hands.HAND_CONNECTIONS,  
                mp_styles.get_default_hand_landmarks_style(), 
                mp_styles.get_default_hand_connections_style() 
            )
        
        # Use first hand for drawing controls
        hand_landmarks = results.multi_hand_landmarks[0] 
        index_finger_tip = hand_landmarks.landmark[8]  # Index finger tip 
        thumb_tip = hand_landmarks.landmark[4]        # Thumb tip
        middle_tip = hand_landmarks.landmark[12]      # Middle finger tip
        
        x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h) 
        thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
        middle_x, middle_y = int(middle_tip.x * w), int(middle_tip.y * h)
        drawing_hand_detected = True

        # Calculate distances for gestures
        reset_distance = math.sqrt((thumb_x - middle_x)**2 + (thumb_y - middle_y)**2)
        draw_distance = math.sqrt((thumb_x - x)**2 + (thumb_y - y)**2)
        
        # Check for reset pinch gesture (thumb and middle finger)
        if reset_distance < 30:  # Pinch threshold
            if not pinch_reset_active:
                canvas = np.zeros((h, w, 3), dtype=np.uint8)
                pinch_reset_active = True
            pinch_reset_detected = True
        else:
            pinch_reset_active = False
            
        # Check for draw pinch gesture (thumb and index finger)
        if draw_distance < 30:
            if not drawing_pinch_active:
                prev_x, prev_y = x, y  # Start new drawing segment
                drawing_pinch_active = True
            pinch_draw_detected = True
        else:
            drawing_pinch_active = False

        # Get current button selection
        current_selection = get_color_selection(x, y) 

        # Handle button interactions
        if current_selection == "reset" and prev_selection != "reset":
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
        elif current_selection == "erase" and prev_selection != "erase":
            erase_mode = not erase_mode  # Toggle erase mode
        elif current_selection in color_palette:
            color = color_palette[current_selection]
            erase_mode = False  # Turn off erase when selecting color

        # Handle drawing/erasing (only when draw pinch is active)
        if drawing_pinch_active and y > 70:  # Changed from 60 to 70
            if prev_x is not None and prev_y is not None:
                if erase_mode:
                    cv2.circle(canvas, (x, y), 30, (0, 0, 0), -1)
                else:
                    cv2.line(canvas, (prev_x, prev_y), (x, y), color, brush_size, lineType=cv2.LINE_AA)
            prev_x, prev_y = x, y
        
    else:
        # Reset positions when no hand detected
        prev_x, prev_y = None, None
        pinch_reset_active = False
        drawing_pinch_active = False

    # Update previous selection state
    prev_selection = current_selection if drawing_hand_detected else None

    # Merge canvas with frame for better visual clarity 
    display_frame = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0) 
    
    # Draw color palette boxes
    cv2.rectangle(display_frame, (50, 10), (100, 50), (255, 0, 0), -1)  # Blue
    cv2.rectangle(display_frame, (110, 10), (160, 50), (0, 255, 0), -1)  # Green
    cv2.rectangle(display_frame, (170, 10), (220, 50), (0, 0, 255), -1)  # Red
    cv2.rectangle(display_frame, (230, 10), (280, 50), (0, 255, 255), -1)  # Yellow
    cv2.rectangle(display_frame, (290, 10), (340, 50), (255, 0, 255), -1)  # Purple
    cv2.rectangle(display_frame, (350, 10), (400, 50), (255, 255, 255), -1)  # White
    
    # Larger reset and erase boxes with centered text below
    # Reset button
    cv2.rectangle(display_frame, (450, 10), (530, 60), (50, 50, 180), -1)  # Taller box (50px height)
    reset_text_x = 490 - cv2.getTextSize("RESET", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0] // 2
    cv2.putText(display_frame, "RESET", (reset_text_x, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Erase button
    cv2.rectangle(display_frame, (550, 10), (630, 60), (100, 100, 100), -1)  # Taller box (50px height)
    eraser_text_x = 590 - cv2.getTextSize("ERASER", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0] // 2
    cv2.putText(display_frame, "ERASER", (eraser_text_x, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Add thin borders to all buttons
    buttons = [
        (50, 10, 100, 50), (110, 10, 160, 50), (170, 10, 220, 50),
        (230, 10, 280, 50), (290, 10, 340, 50), (350, 10, 400, 50),
        (450, 10, 530, 60), (550, 10, 630, 60)  # Updated for taller buttons
    ]
    for x1, y1, x2, y2 in buttons:
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (200, 200, 200), 1)

    # Display the frames
    cv2.imshow("Air Canvas", display_frame) 
    
    # Create canvas display (black background)
    canvas_display = np.zeros((600, 800, 3), dtype=np.uint8)
    canvas_resized = cv2.resize(canvas, (800, 600))
    canvas_display = cv2.addWeighted(canvas_display, 0.2, canvas_resized, 0.8, 0)
    cv2.imshow("Canvas", canvas_display)
 
    key = cv2.waitKey(1) & 0xFF 
    if key == ord('q') or key == 27:  # 27 is ESC key
        break

cap.release() 
cv2.destroyAllWindows()