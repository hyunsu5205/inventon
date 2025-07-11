import cv2
import numpy as np
import sys
import signal
from picamera2 import Picamera2

# 전역 변수
picam2 = None

def signal_handler(sig, frame):
    """Ctrl+C 신호 처리"""
    print("\n프로그램을 종료합니다...")
    if picam2 is not None:
        picam2.stop()
        picam2.close()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# 모델 설정
model_name = 'res10_300x300_ssd_iter_140000.caffemodel'
prototxt_name = 'deploy.prototxt.txt'
min_confidence = 0.5

def detectAndDisplay(frame, frame_count, model):
    """DNN을 사용한 실시간 얼굴 인식"""
    (height, width) = frame.shape[:2]
    
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0))

    model.setInput(blob)
    detections = model.forward()

    face_count = 0
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > min_confidence:
            face_count += 1
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int")
            
            print(f"프레임 {frame_count} - 얼굴 {face_count}: 신뢰도 {confidence*100:.1f}%, 위치({startX},{startY})-({endX},{endY})")
    
    if face_count > 0:
        print(f"프레임 {frame_count}: 총 {face_count}개 얼굴 발견")
    
    return face_count

print("🎥 PiCamera2 + OpenCV DNN 실시간 얼굴 인식")
print("=" * 50)
print(f"OpenCV 버전: {cv2.__version__}")

# 모델 로드
print("DNN 모델을 로드중...")
try:
    model = cv2.dnn.readNetFromCaffe(prototxt_name, model_name)
    print("✅ 모델 로드 완료!")
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")
    exit(1)

# PiCamera2 초기화
print("PiCamera2를 초기화중...")
try:
    picam2 = Picamera2()
    
    # 카메라 설정
    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam2.configure(config)
    
    print("✅ PiCamera2 초기화 완료!")
    print(f"해상도: 640x480")
    print(f"최소 신뢰도 임계값: {min_confidence}")
    
    # 카메라 시작
    picam2.start()
    print("카메라가 시작되었습니다...")
    print("실시간 얼굴 인식을 시작합니다. Ctrl+C로 종료하세요.")
    print("-" * 50)
    
except Exception as e:
    print(f"❌ PiCamera2 초기화 실패: {e}")
    exit(1)

frame_count = 0
total_faces = 0

try:
    while True:
        # 프레임 캡처
        frame = picam2.capture_array()
        
        if frame is None:
            print("프레임을 읽을 수 없습니다.")
            break
            
        frame_count += 1
        
        # 3프레임마다 한 번씩 얼굴 인식 (성능 최적화)
        if frame_count % 3 == 0:
            face_count = detectAndDisplay(frame, frame_count, model)
            total_faces += face_count
            
        # 100프레임마다 통계 출력
        if frame_count % 100 == 0:
            print(f"\n📊 통계 (프레임 {frame_count}): 총 {total_faces}개 얼굴 누적 발견")
            print("-" * 50)
            
except KeyboardInterrupt:
    print(f"\n키보드 인터럽트로 종료합니다.")
    print(f"📊 최종 통계: {frame_count}프레임에서 총 {total_faces}개 얼굴 발견")
except Exception as e:
    print(f"오류 발생: {e}")
finally:
    if picam2 is not None:
        picam2.stop()
        picam2.close()
    print("✅ 리소스 정리 완료!") 