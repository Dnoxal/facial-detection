import cv2
import dlib
import imutils
import numpy as np
from imutils import face_utils

# comment for self : make sure to use python /Users/danielli/Desktop/tracking/main.py 
# to run within console since using anaconda environment

VIDEO_PATH = "test.mp4" #this should be within the same folder as main.py, test video given by dr. coronel
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
TARGET_WIDTH = 960    
UPSAMPLE = 1             # for dlib detector

def visualize_facial_landmarks(image, shape, alpha=0.75):
    palette = [
        (19, 199, 109), (79, 76, 240), (230, 159, 23),
        (168, 100, 168), (158, 163, 32), (163, 38, 32), (180, 42, 220)
    ]
    overlay = image.copy()
    output = image.copy()

    regions = list(face_utils.FACIAL_LANDMARKS_IDXS.keys())
    for i, name in enumerate(regions):
        color = palette[i % len(palette)]
        (j, k) = face_utils.FACIAL_LANDMARKS_IDXS[name]
        pts = shape[j:k]

        if name == "jaw":
            for l in range(1, len(pts)):
                ptA = tuple(pts[l - 1])
                ptB = tuple(pts[l])
                cv2.line(overlay, ptA, ptB, color, 2)
        else:
            hull = cv2.convexHull(pts)
            cv2.drawContours(overlay, [hull], -1, color, -1)

    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    return output

def main():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter("test_annotated.mp4", fourcc, fps, (W, H))

    # CSV
    import csv
    csv_file = open("test_faces.csv", "w", newline="")
    wcsv = csv.writer(csv_file)
    wcsv.writerow(["frame", "x", "y", "w", "h", "landmarks"])  

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        resized = imutils.resize(frame, width=TARGET_WIDTH) if TARGET_WIDTH and W > 0 else frame
        rH, rW = resized.shape[:2]
        sx = W / float(rW)
        sy = H / float(rH)

        gray_small = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        rects = detector(gray_small, UPSAMPLE)

        if rects:
            # pick largest
            rect = max(rects, key=lambda rr: (rr.right() - rr.left()) * (rr.bottom() - rr.top()))
            # scale rect to full-res
            x1 = int(rect.left() * sx);  y1 = int(rect.top() * sy)
            x2 = int(rect.right() * sx); y2 = int(rect.bottom() * sy)

            # landmarks on full-res
            gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rect_full = dlib.rectangle(x1, y1, x2, y2)
            shape = predictor(gray_full, rect_full)
            shape_np = face_utils.shape_to_np(shape)

            annotated = visualize_facial_landmarks(frame.copy(), shape_np)

            # stable face bbox from landmarks
            fx, fy, fw, fh = cv2.boundingRect(np.array([shape_np]))
            cv2.rectangle(annotated, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)

            writer.write(annotated)
            wcsv.writerow([frame_idx, fx, fy, fw, fh, 68])
        else:
            writer.write(frame)
            wcsv.writerow([frame_idx, "", "", "", "", 0])

        frame_idx += 1

    cap.release()
    writer.release()
    csv_file.close()
    print("works")

if __name__ == "__main__":
    main()
