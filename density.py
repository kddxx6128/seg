import cv2
import numpy as np
from ultralytics import YOLO


def main():
    model = YOLO("yolo26n-seg.pt")
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("에러: 카메라를 열 수 없습니다.")
        return

    print("인파 밀집 모니터링 시스템 가동 중... 종료 -> 'q'")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        results = model(frame, classes=[0], conf=0.4, verbose=False)
        annotated_frame = results[0].plot()
        person_count = len(results[0].boxes)
        density_percent = 0.0

        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()

            if len(masks) > 0:
                combined_mask = np.max(masks, axis=0) > 0.5
                density_percent = (np.sum(combined_mask) / combined_mask.size) * 100

        if person_count >= 5 and density_percent >= 30.0:
            status = "DANGER"
            color = (0, 0, 255)
        elif person_count >= 3 or density_percent >= 15.0:
            status = "WARNING"
            color = (0, 255, 255)
        else:
            status = "NORMAL"
            color = (0, 255, 0)

        cv2.rectangle(annotated_frame, (0, 0), (640, 140), (0, 0, 0), -1)

        cv2.putText(
            annotated_frame,
            f"PEOPLE: {person_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            annotated_frame,
            f"DENSITY : {density_percent:.1f}%",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            annotated_frame,
            f"STATUS        : {status}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )

        if status == "DANGER":
            cv2.rectangle(
                annotated_frame,
                (0, 0),
                (frame.shape[1], frame.shape[0]),
                (0, 0, 255),
                8,
            )

        cv2.imshow("군중 밀집 감지", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
