import cv2
import os

def detect_ambulances(video_path, template_paths):
    templates = []
    # Load all templates into a list
    for template_path in template_paths:
        template = cv2.imread(template_path, 0)  # Load in grayscale
        if template is not None:
            templates.append((template, template.shape[:2]))
        else:
            print(f"Error: Unable to load template {template_path}")
    if not templates:
        print("Error: No valid templates loaded.")
        return
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video.")
        return
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for template, (template_height, template_width) in templates:
            # Check if the template is larger than the frame
            if template.shape[0] > gray_frame.shape[0] or template.shape[1] > gray_frame.shape[1]:
                print(f"Skipping template {template.shape} as it is larger than frame {gray_frame.shape}.")
                continue

            # Perform template matching
            match_result = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match_result)
            # Threshold for a match
            threshold = 0.6
            if max_val >= threshold:
                top_left = max_loc
                bottom_right = (top_left[0] + template_width, top_left[1] + template_height)
                # Draw a rectangle around the match
                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
                cv2.putText(frame, f"Ambulance {max_val:.2f}", (top_left[0], top_left[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
        # Display the frame with matches
        cv2.imshow('Ambulance Finder', frame)

        # Press 'q' to exit the video playback
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
# File paths
video_path = r"C:\Users\Admin\Desktop\ambulance_finder1\ambulance_finder\Ambulance rams bike riders at traffic signal.mp4"  # Replace with your video file path
template_folder = r"C:\Users\Admin\Desktop\ambulance_finder1\ambulance_finder\ambulance"  # Replace with the folder containing template images
template_paths = [os.path.join(template_folder, f) for f in os.listdir(template_folder) if f.endswith('.jpg') or f.endswith('.png')]
# Call the function
detect_ambulances(video_path, template_paths)

