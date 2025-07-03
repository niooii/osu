import time
import numpy as np
import cv2
from enum import Enum
import math

FADE_TIME = 1300

class HitEvent(Enum):
    ZERO = 1
    FIFTY = 2
    HUNDRED = 3

# (x, y, event): time in ms 
already_saw_events: dict[tuple[int, int, HitEvent], int] = {}

def average_color_circle(image, center_x, center_y, radius):
    image_shape = image.shape[:2]
    y_coords, x_coords = np.ogrid[:image_shape[0], :image_shape[1]]
    distances_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    
    # circular mask
    mask = distances_from_center <= radius
    
    pixels_in_circle = image[mask]
    
    if len(pixels_in_circle) > 0:
        mean_blue = np.mean(pixels_in_circle[:, 0])
        mean_green = np.mean(pixels_in_circle[:, 1]) 
        mean_red = np.mean(pixels_in_circle[:, 2])
        
        return (int(mean_red), int(mean_green), int(mean_blue))
    else:
        return (0, 0, 0)  

def extract_events(frame: np.ndarray) -> list[tuple[int, int, int, HitEvent]]:
    events = []
    
    # find circles
    dp=1
    min_dist=30
    param1=50
    param2=45
    min_radius=8
    max_radius=80
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 1)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        
        for (x, y, rad) in circles:
            x = math.floor(x)
            y = math.floor(y)
            rad = math.floor(rad)
            
            # Check if the circle is within image bounds
            if (x - rad > 0 and y - rad > 0 and 
                x + rad < frame.shape[1] and y + rad < frame.shape[0]):
                
                avg_colors_rgb = average_color_circle(frame, x, y, rad)

                max_col_idx = avg_colors_rgb.index(max(avg_colors_rgb))

                r, g, b = avg_colors_rgb

                sorted_colors = sorted(avg_colors_rgb)

                # skip if numbers are all close to each other
                epsilon = 5
                
                # standard deviation 
                stddev = math.sqrt(sum((x - np.average(sorted_colors)) ** 2 for x in sorted_colors) / len(sorted_colors))
                if stddev < epsilon:
                    continue

                # skip if the color is too close to black
                if np.average(avg_colors_rgb) < 80:
                    continue

                tmp_event = None
                yellow = (r + g) / 2
                if yellow > 150:
                    # test the ratio of blue to yellow
                    # if it is mostly yellow then its a 50
                    tmp_event = HitEvent.FIFTY
                elif max_col_idx == 1:
                    # if it is mostly green then its a 100
                    tmp_event = HitEvent.HUNDRED
                elif max_col_idx == 0:
                    # if it is mostly red then its a 0
                    tmp_event = HitEvent.ZERO
                
                if tmp_event is not None and (x, y, tmp_event) not in already_saw_events:
                    events.append((x, y, rad, tmp_event))
                    already_saw_events[(x, y, tmp_event)] = int(time.time() * 1000)
        
    # remove events that are too old
    old_keys = [k for k, v in already_saw_events.items() if time.time() * 1000 - v > FADE_TIME]
    for key in old_keys:
        del already_saw_events[key]

    return events

def on_frame(frame: np.ndarray):
    
    # Extract and process all events
    events = extract_events(frame)
    for (x, y, r, event) in events:
        cv2.circle(frame, (x, y), r, (255, 255, 255), 2)
        cv2.putText(frame, event.name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        print(f"Event detected: {event.name} at ({x}, {y})")
    
    cv2.imshow("Game View", frame)
    cv2.waitKey(1)
