import numpy as np
import cv2

#Define face detector class for portability and room for expansion
class FaceDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        self.frame_gray = None
        
    def detectFaceLocation(self,frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(frame_gray, 1.3, 8)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]  # First detected face
            return (x,y,w,h)
        else:
            return None

    def detect_face_single(self,frame):
        self.frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.frame_gray = self.frame_gray.astype(np.uint8)  # Just in case
        self.frame_gray = cv2.equalizeHist(self.frame_gray)
        faces = self.face_cascade.detectMultiScale(self.frame_gray, 1.3, 5)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]  # First detected face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        return frame

class DollyZoom:
    def __init__(self, cap, face_detector, transform_matrixes):
        self.cap = cap
        ret, frame = self.cap.read()
        self.first_frame = frame
        self.face_detector = face_detector
        self.webcam_shape = frame.shape
        self.webcam_h, self.webcam_w, self.webcam_d = frame.shape
        self.aspect_ratio = 2.39
        #Initialize transform matrixes
        self.transform_matrixes = transform_matrixes
        print(f'Transform shape:{self.transform_matrixes.shape}')
        #Adjust transforms to find transforms from original frame
        transform_list = []
        transform_list.append(np.eye(3))
        for counter in range(self.transform_matrixes.shape[2]):
            transform_list.append(transform_list[counter] @ self.transform_matrixes[:,:,counter])
        self.transform_matrixes_combined = np.stack(transform_list,axis=0)
        print(f'Transform combined shape:{self.transform_matrixes_combined.shape}')
        # transform_matrixes *= 1.5

        # Initialize frame detection info
        self.calculation_frame = None
        self.viewing_frame = None
        self.face_info = (0,0,self.webcam_w,self.webcam_h)

        # Homography flags
        self.user_ready = False
        self.transform_count = 0

        # Smoothing variables
        self.smooth_cx = 0
        self.smooth_cy = 0
        self.smooth_w = 0
        self.smooth_h = 0
        self.alpha = 0.1

        # Motion rejection
        self.last_box = None
        self.distance_rejection_threshold = 50

    def read_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.frame = frame
        return ret

    def detect_face(self):
        if self.frame is not None:
            self.face_info = self.face_detector.detectFaceLocation(self.frame)

    def update_calculation_frame(self):
        if self.frame is None or self.face_info is None:
            return
        
        x, y, w, h = self.face_info

        face_cx = x + w // 2
        face_cy = y + h // 2
        face_w = w

        if self.smooth_cx is None:
            self.smooth_cx = face_cx
            self.smooth_cy = face_cy
            self.smooth_w = face_w

        self.smooth_cx = self.alpha * face_cx + (1 - self.alpha) * self.smooth_cx
        self.smooth_cy = self.alpha * face_cy + (1 - self.alpha) * self.smooth_cy
        self.smooth_w = self.alpha * face_w + (1 - self.alpha) * self.smooth_w

        face_cx = int(self.smooth_cx)
        face_cy = int(self.smooth_cy)
        face_w = int(self.smooth_w)

        current_box = np.array([face_cx, face_cy, face_w])
        if self.last_box is None:
            self.last_box = current_box
        else:
            dist = np.linalg.norm(current_box - self.last_box)
            if dist > self.distance_rejection_threshold:
                self.last_box = current_box
            else:
                current_box = self.last_box

        face_cx, face_cy, face_w = map(int, current_box)

        target_w = int(face_w * 3 * 1.2)
        target_h = int(target_w / self.aspect_ratio)

        x1 = max(face_cx - target_w // 2, 0)
        y1 = max(face_cy - target_h // 2, 0)
        x2 = min(x1 + target_w, self.webcam_w)
        y2 = min(y1 + target_h, self.webcam_h)

        x1 = max(x2 - target_w, 0)
        y1 = max(y2 - target_h, 0)

        self.calculation_frame = self.frame[y1:y2, x1:x2].copy()
        self.calculation_frame = self.fit_frame(self.calculation_frame)

    def viewing_frame_normal(self):
        if self.calculation_frame is None:
            return
        calc_h, calc_w = self.calculation_frame.shape[:2]
        view_w = int(calc_w / 1.2)
        view_h = int(calc_h / 1.2)
        vx1 = (calc_w - view_w) // 2
        vy1 = (calc_h - view_h) // 2
        vx2 = vx1 + view_w
        vy2 = vy1 + view_h
        self.viewing_frame = self.calculation_frame[vy1:vy2, vx1:vx2].copy()

    def fit_frame(self,frame):
        if frame is not None:
            h, w, d = frame.shape
            scale = self.webcam_w/ w
            new_h = int((h)*scale)
            scaled = cv2.resize(frame,(self.webcam_w,new_h))
            return scaled
        else:
            self.read_frame()
            return self.frame

    def get_calculation_frame(self):
        return self.calculation_frame

    def get_viewing_frame(self):
        return self.viewing_frame

    def warp_calculation_frame(self, counter):
        if self.calculation_frame is None:
            return
        
        h, w = self.calculation_frame.shape[:2]
        
        # Ensure the counter is within bounds
        if counter < 0 or counter >= self.transform_matrixes.shape[2]:
            print(f"Invalid counter: {counter}")
            return

        #H = self.transform_matrixes_combined[counter, :, :]
        H = self.transform_matrixes[:, :, counter]
        warped = cv2.warpPerspective(self.calculation_frame, H, (w, h))
        self.calculation_frame = warped

    def get_flipped_view(self):
        view = self.get_viewing_frame()
        if view is not None and view.size != 0:
            return np.flip(view, axis=1)
        return None

doVideoRecording = True #Set this to record

#Use webcam
cap = cv2.VideoCapture(0)

if doVideoRecording:
    #Get codec and videowriter objct
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('project.mp4',codec,30,(640,480))

face_detector = FaceDetector()

#Load transformation matrixes
transform_matrixes = np.load('dolly_zoom_v3.npy')

webcam = DollyZoom(cap=cap,face_detector=face_detector,transform_matrixes=transform_matrixes)

counter = 0
warp_mode = False
while True:
    webcam.read_frame()
    webcam.detect_face()
    webcam.update_calculation_frame()

    if warp_mode:
        if counter < webcam.transform_matrixes_combined.shape[0]:
            webcam.warp_calculation_frame(counter)
            counter += 1
            print(counter)
        else:
            warp_mode = False  # Done with all transforms

    webcam.viewing_frame_normal()
    # shown_frame = webcam.get_flipped_view()
    shown_frame = webcam.get_viewing_frame()

    if shown_frame is not None:
        cv2.imshow('Press Spacebar when ready', shown_frame)
        if doVideoRecording:
            # Pad to at least the target size
            height, width = shown_frame.shape[:2]
            top = max((480 - height) // 2, 0)
            bottom = max(480 - height - top, 0)
            left = max((640 - width) // 2, 0)
            right = max(640 - width - left, 0)

            # Pad with black borders
            padded_frame = cv2.copyMakeBorder(
                shown_frame, top, bottom, left, right,
                borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )

            # Crop in case it's now too big
            padded_frame = padded_frame[:480, :640]
            out.write(padded_frame)
    else:
        print('issue showing frame')

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' ') and not warp_mode:
        # Trigger the warp sequence
        warp_mode = True
        counter = 0



# Close the window / Release webcam
cap.release()
if doVideoRecording:
    # After we release our webcam, we also release the output
    out.release() 
# De-allocate any associated memory usage 
cv2.destroyAllWindows()