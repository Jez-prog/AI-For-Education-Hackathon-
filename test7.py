import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, 
                      min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def recognize_asl(landmarks):
    tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
    fingers = []
    
    # Thumb detection - ASL specific
    thumb_tip = landmarks[tips[0]]
    thumb_ip = landmarks[tips[0]-1]  # Thumb interphalangeal joint
    fingers.append(1 if thumb_tip.x < thumb_ip.x else 0)  # Thumb extended=1
    
    for i in range(1, 5):
        tip = landmarks[tips[i]]
        pip = landmarks[tips[i]-2]  
        fingers.append(1 if tip.y < pip.y else 0)  
    if (fingers == [0, 0, 0, 0, 0] and 
        abs(thumb_tip.x - landmarks[5].x) < 0.05):  
        return "E"
    
    elif fingers == [1, 0, 0, 0, 0]:
        return "A"
    
   
    elif fingers == [0, 1, 1, 1 , 1]:
        return "B"
    
    elif all(fingers):  # All fingers extended
        if len(landmarks) >= 21 and hasattr(landmarks[0], 'z'):
            index_tip = landmarks[8]
        index_mcp = landmarks[5]
        pinky_tip = landmarks[20]
        pinky_mcp = landmarks[17]

        # Curve in depth (Z axis)
        index_z_curve = index_mcp.z - index_tip.z
        pinky_z_curve = pinky_mcp.z - pinky_tip.z

        # Horizontal width between index and pinky
        hand_width = abs(index_tip.x - pinky_tip.x)

        # C shape sideways: fingers curled forward, and moderate width
        if index_z_curve > 0.05 and pinky_z_curve > 0.05 and 0.10 < hand_width < 0.25:
            return "C"



    
    elif fingers == [1, 1, 0, 0, 0] and landmarks[8].y < landmarks[6].y:
        return "D"
    
    elif fingers == [1, 1, 0, 0, 1]:
        return "Love You"

    elif (fingers == [0, 0, 1, 1, 1] and  # Thumb and index folded, others extended
      abs(landmarks[4].y - landmarks[8].y) < 0.05 and  # Thumb and index tips close vertically
      abs(landmarks[4].x - landmarks[8].x) < 0.05):    # Thumb and index tips close horizontally
        return "F"
    elif fingers == [1, 0, 1, 0, 0]:
        return "Stormie"
    
    else:
        return "?"

cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue
    
    frame = cv2.flip(frame, 1)
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255,255,225), thickness=2),  # Yellow landmarks
                mp_drawing.DrawingSpec(color=(255,255,255), thickness=2)   # Cyan connections
            )
            
            sign = recognize_asl(hand_landmarks.landmark)
            cx, cy = int(hand_landmarks.landmark[0].x * frame.shape[1]), \
                     int(hand_landmarks.landmark[0].y * frame.shape[0])
            
            cv2.putText(frame, sign, (cx-20, cy-50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)
    
    cv2.imshow('ASL: A,C,D,E Detection', frame)
    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()