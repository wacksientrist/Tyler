import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

global X_OFF, Y_OFF, Z_OFF, dot_position_3d

# Constants
X_OFF = 0  # X Offset
Y_OFF = 0  # Y Offset
Z_OFF = 0  # Z Offset
BUFFER_SIZE = 5  # Number of positions to average
DOT_RADIUS = 10  # Radius of the dot to draw
SCALE_FACTOR = 1  # Scale factor to fit the dot in the plot area
MOVEMENT_THRESHOLD = 5  # Minimum pixel movement to be considered a valid movement

# Global buffers for averaging positions
positions_buffer1 = []
positions_buffer2 = []
previous_position1 = None
previous_position2 = None
template = None
template_hsv = None
template_height = 0
template_width = 0

def select_template(frame):
    global template, template_hsv, template_height, template_width
    # Let the user select a region in the frame to use as the template
    r = cv2.selectROI("Select Template", frame, fromCenter=False, showCrosshair=True)
    x, y, w, h = r
    template = frame[y:y+h, x:x+w]
    template_hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)  # Convert to HSV
    template_height, template_width = template.shape[:2]
    cv2.destroyWindow("Select Template")

def track_template(frame):
    global template_hsv, template_height, template_width
    if template_hsv is not None:
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create a mask for the template's color
        lower_color = np.array([template_hsv[..., 0].min() - 10, 100, 100])
        upper_color = np.array([template_hsv[..., 0].max() + 10, 255, 255])
        mask = cv2.inRange(frame_hsv, lower_color, upper_color)
        
        # Use the mask with the template matching
        result = cv2.matchTemplate(mask, cv2.cvtColor(template, cv2.COLOR_BGR2GRAY), cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(result)
        return (max_loc[0] + template_width // 2, max_loc[1] + template_height // 2)  # Return center of the template
    return None

def triangulate_point(pt1, pt2):
    if pt1 is None or pt2 is None:
        return None

    x1, y1 = pt1
    x2, y2 = pt2

    y = x1
    x = -x2  # Adjust for camera flipping
    z = -((y1 + y2) / 2)  # Adjust for camera flipping

    y += Y_OFF
    x += X_OFF
    z += Z_OFF

    return (x, y, z)

def average_positions(buffer):
    if len(buffer) == 0:
        return None
    x = np.mean([pos[0] for pos in buffer])
    y = np.mean([pos[1] for pos in buffer])
    return (int(x), int(y))

def draw_dot(frame, position):
    if position is not None:
        cv2.circle(frame, position, DOT_RADIUS, (0, 255, 0), -1)  # Draw a green dot

def update_3d_plot(dot_position_3d):
    ax.cla()  # Clear the current axes
    if dot_position_3d is not None:
        scaled_x = dot_position_3d[0] * SCALE_FACTOR
        scaled_y = dot_position_3d[1] * SCALE_FACTOR
        scaled_z = dot_position_3d[2] * SCALE_FACTOR

        ax.scatter(scaled_x, scaled_y, scaled_z, c='r', s=100)  # Draw the dot as a sphere
        ax.set_xlim([0, 3])
        ax.set_ylim([0, 3])
        ax.set_zlim([0, 3])
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
    plt.draw()
    plt.pause(0.01)

# Setup the 3D plot
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('black')

def scale_value(value, old_range, new_range):
    return np.interp(value, old_range, new_range)

old_range = (-500, 2000)
new_range = (0, 3)

def main():
    global dot_position_3d
    cap1 = cv2.VideoCapture(0)  # Camera 1
    cap2 = cv2.VideoCapture(1)  # Camera 2

    # Read the first frame for template selection
    ret1, frame1 = cap1.read()
    if ret1:
        select_template(frame1)

    with ThreadPoolExecutor() as executor:
        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if not ret1 or not ret2:
                break

            dot_position1 = track_template(frame1)
            dot_position2 = track_template(frame2)

            # Check for movement
            if dot_position1 is not None and previous_position1 is not None:
                distance1 = np.linalg.norm(np.array(dot_position1) - np.array(previous_position1))
                if distance1 < MOVEMENT_THRESHOLD:
                    dot_position1 = None  # Ignore if the movement is too small

            if dot_position2 is not None and previous_position2 is not None:
                distance2 = np.linalg.norm(np.array(dot_position2) - np.array(previous_position2))
                if distance2 < MOVEMENT_THRESHOLD:
                    dot_position2 = None  # Ignore if the movement is too small

            # Store current positions for the next frame
            previous_position1 = dot_position1
            previous_position2 = dot_position2

            # Add the new positions to the buffer
            if dot_position1 is not None:
                positions_buffer1.append(dot_position1)
                if len(positions_buffer1) > BUFFER_SIZE:
                    positions_buffer1.pop(0)

            if dot_position2 is not None:
                positions_buffer2.append(dot_position2)
                if len(positions_buffer2) > BUFFER_SIZE:
                    positions_buffer2.pop(0)

            # Average the positions
            averaged_position1 = average_positions(positions_buffer1)
            averaged_position2 = average_positions(positions_buffer2)

            # Find the 3D position
            dot_position_3d = triangulate_point(averaged_position1, averaged_position2)

            if dot_position_3d:
                dot_position_3d = (
                    float(scale_value(dot_position_3d[0], old_range, new_range)),
                    float(scale_value(dot_position_3d[1], old_range, new_range)),
                    float(scale_value(dot_position_3d[2], old_range, new_range))
                )

            # Draw the averaged position on the camera frames
            draw_dot(frame1, averaged_position1)
            draw_dot(frame2, averaged_position2)

            if dot_position_3d:
                print(f"Averaged 3D Dot Position: {dot_position_3d}")

            cv2.imshow("Camera 1", frame1)
            cv2.imshow("Camera 2", frame2)

            # Update the 3D plot
            update_3d_plot(dot_position_3d)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
