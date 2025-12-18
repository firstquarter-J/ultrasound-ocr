# 태아 초음파 영상 OCR 프로젝트 구현 계획

## 📅 프로젝트 타임라인

**전체 기간**: 약 2-3주 (MVP 기준)
**목표 버전**: v0.1 MVP → v0.2 고급 기능 → v0.3 의료 특화 → v1.0 프로덕션

---

## Phase 1: 기본 인프라 구축 (1-2일)

### 목표
프로젝트의 기본 골격을 구축하고 개발 환경을 설정합니다.

### 작업 항목

#### 1.1 의존성 관리
- [ ] `requirements.txt` 작성
  - 핵심 라이브러리: easyocr, opencv-python, numpy, pillow
  - 유틸리티: pyyaml, loguru, python-dotenv
  - 버전 고정으로 재현 가능성 확보

- [ ] `requirements-dev.txt` 작성
  - 테스트: pytest, pytest-cov
  - 코드 품질: black, flake8, mypy
  - 개발 도구: jupyter, ipython

- [ ] 가상환경 설정 및 의존성 설치 검증
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  pip install -r requirements-dev.txt
  ```

#### 1.2 설정 파일 시스템
- [ ] `configs/default.yaml` 생성
  ```yaml
  project:
    name: "ultrasound-ocr"
    version: "0.1.0"

  logging:
    level: "INFO"
    format: "{time} | {level} | {message}"
    output_dir: "logs"

  preprocessing:
    default_method: "clahe"

  ocr:
    default_engine: "easyocr"
    confidence_threshold: 0.6
  ```

- [ ] `configs/preprocessing.yaml` 생성 (전처리 파라미터 상세 설정)
- [ ] `configs/ocr_engines.yaml` 생성 (OCR 엔진별 설정)

#### 1.3 유틸리티 모듈 구현
- [ ] `src/utils/logger.py` - Loguru 기반 로깅 시스템
  ```python
  from loguru import logger

  def setup_logger(config):
      logger.add(
          f"{config['output_dir']}/app.log",
          rotation="10 MB",
          level=config['level']
      )
      return logger
  ```

- [ ] `src/utils/config.py` - YAML 설정 로드
  ```python
  import yaml

  class Config:
      def __init__(self, config_path):
          with open(config_path, 'r') as f:
              self.data = yaml.safe_load(f)

      def get(self, key, default=None):
          return self.data.get(key, default)
  ```

- [ ] `src/utils/metrics.py` - 성능 메트릭 수집 (기본 구조)

#### 1.4 환경 설정
- [ ] `.env.example` 생성 (환경 변수 템플릿)
  ```
  EASYOCR_MODEL_PATH=~/.EasyOCR/model
  TESSERACT_PATH=/usr/local/bin/tesseract
  GPU_ENABLED=true
  ```

- [ ] `pytest.ini` 생성 (테스트 설정)
  ```ini
  [pytest]
  testpaths = tests
  python_files = test_*.py
  python_functions = test_*
  addopts = -v --strict-markers
  ```

- [ ] `setup.py` 생성 (패키지 설정)

### 검증 기준
✅ 가상환경에서 모든 의존성 설치 성공
✅ 설정 파일 로드 및 파싱 정상 동작
✅ 로거가 파일 및 콘솔 출력 정상 수행
✅ 기본 유틸리티 모듈 import 오류 없음

---

## Phase 2: 핵심 OCR 기능 구현 (3-4일)

### 목표
EasyOCR을 통합하고 단일 이미지에서 텍스트를 추출하는 기본 기능을 완성합니다.

### 작업 항목

#### 2.1 OCR 인터페이스 설계
- [ ] `src/ocr/base.py` - 추상 OCR 엔진 인터페이스
  ```python
  from abc import ABC, abstractmethod
  from typing import Dict, List, Tuple

  class BaseOCREngine(ABC):
      @abstractmethod
      def recognize(self, image) -> Dict:
          """
          이미지에서 텍스트 인식

          Args:
              image: numpy array 또는 PIL Image

          Returns:
              {
                  'text': str,
                  'confidence': float,
                  'boxes': List[Tuple],  # 바운딩 박스
                  'details': List[Dict]  # 상세 결과
              }
          """
          pass

      @abstractmethod
      def load_model(self):
          """모델 로드"""
          pass
  ```

#### 2.2 EasyOCR 엔진 구현
- [ ] `src/ocr/easyocr_engine.py` 구현
  ```python
  import easyocr
  from .base import BaseOCREngine

  class EasyOCREngine(BaseOCREngine):
      def __init__(self, config):
          self.config = config
          self.reader = None
          self.load_model()

      def load_model(self):
          self.reader = easyocr.Reader(
              lang_list=self.config['languages'],
              gpu=self.config['gpu']
          )

      def recognize(self, image):
          results = self.reader.readtext(image)

          text = ' '.join([res[1] for res in results])
          confidence = sum([res[2] for res in results]) / len(results) if results else 0

          return {
              'text': text,
              'confidence': confidence,
              'boxes': [res[0] for res in results],
              'details': results
          }
  ```

- [ ] GPU 가용성 확인 및 폴백 로직 추가
- [ ] 모델 다운로드 및 캐싱 처리

#### 2.3 기본 전처리 구현
- [ ] `src/preprocessing/image_enhancer.py` - 기본 전처리
  ```python
  import cv2
  import numpy as np

  class ImageEnhancer:
      def __init__(self, config):
          self.config = config

      def enhance(self, image):
          """기본 이미지 향상"""
          # 그레이스케일 변환
          if len(image.shape) == 3:
              gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
          else:
              gray = image

          # CLAHE 적용
          clahe = cv2.createCLAHE(
              clipLimit=self.config['clahe']['clip_limit'],
              tileGridSize=tuple(self.config['clahe']['tile_grid_size'])
          )
          enhanced = clahe.apply(gray)

          return enhanced
  ```

- [ ] `src/preprocessing/binarizer.py` - 이진화
  ```python
  def adaptive_threshold(image, config):
      return cv2.adaptiveThreshold(
          image,
          255,
          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
          cv2.THRESH_BINARY,
          config['block_size'],
          config['c']
      )
  ```

#### 2.4 단일 이미지 OCR 스크립트
- [ ] `scripts/run_ocr.py` 구현
  ```python
  import argparse
  import cv2
  from src.ocr.easyocr_engine import EasyOCREngine
  from src.preprocessing.image_enhancer import ImageEnhancer
  from src.utils.config import Config
  from src.utils.logger import setup_logger

  def main():
      parser = argparse.ArgumentParser()
      parser.add_argument('image_path', help='Path to image')
      parser.add_argument('--config', default='configs/default.yaml')
      args = parser.parse_args()

      # 설정 로드
      config = Config(args.config)
      logger = setup_logger(config.get('logging'))

      # 이미지 로드
      image = cv2.imread(args.image_path)

      # 전처리
      enhancer = ImageEnhancer(config.get('preprocessing'))
      enhanced = enhancer.enhance(image)

      # OCR 실행
      ocr = EasyOCREngine(config.get('ocr'))
      result = ocr.recognize(enhanced)

      # 결과 출력
      print(f"텍스트: {result['text']}")
      print(f"신뢰도: {result['confidence']:.2f}")

  if __name__ == '__main__':
      main()
  ```

#### 2.5 결과 저장 포맷
- [ ] JSON 출력 구조 정의 및 구현
  ```python
  import json
  from datetime import datetime

  def save_result(result, output_path):
      output = {
          'timestamp': datetime.now().isoformat(),
          'ocr_results': result,
          'metadata': {
              'engine': 'easyocr',
              'version': '1.7.0'
          }
      }

      with open(output_path, 'w', encoding='utf-8') as f:
          json.dump(output, f, ensure_ascii=False, indent=2)
  ```

### 검증 기준
✅ 샘플 초음파 이미지에서 텍스트 추출 성공
✅ 신뢰도 점수가 0.6 이상인 결과 획득
✅ JSON 형식으로 결과 저장 정상 동작
✅ 로그 파일에 처리 과정 기록 확인

---

## Phase 3: 전처리 고도화 (2-3일)

### 목표
의료 영상 특성에 맞는 고급 전처리 알고리즘을 구현하여 OCR 정확도를 향상시킵니다.

### 작업 항목

#### 3.1 노이즈 제거 구현
- [ ] `src/preprocessing/noise_reducer.py`
  ```python
  class NoiseReducer:
      def bilateral_filter(self, image, d=9, sigma_color=75, sigma_space=75):
          """엣지 보존 노이즈 제거"""
          return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

      def non_local_means(self, image, h=10, template_size=7, search_size=21):
          """Non-local Means Denoising"""
          return cv2.fastNlMeansDenoising(image, None, h, template_size, search_size)
  ```

#### 3.2 고급 이진화 기법
- [ ] Otsu's Method 구현
- [ ] Sauvola 이진화 구현 (의료 영상에 효과적)
- [ ] 다중 이진화 방법 비교 기능

#### 3.3 형태학적 연산
- [ ] Opening (미세 노이즈 제거)
- [ ] Closing (문자 연결)
- [ ] 커널 크기 자동 조정 로직

#### 3.4 전처리 파이프라인 구성
- [ ] 순차적 전처리 체인 구현
  ```python
  class PreprocessingPipeline:
      def __init__(self, config):
          self.steps = [
              ('grayscale', self.to_grayscale),
              ('denoise', NoiseReducer().bilateral_filter),
              ('enhance', ImageEnhancer().clahe),
              ('binarize', Binarizer().adaptive_threshold),
              ('morph', self.morphological_ops)
          ]

      def process(self, image):
          for name, step in self.steps:
              image = step(image)
              logger.info(f"Applied {name}")
          return image
  ```

#### 3.5 실험 노트북 작성
- [ ] `notebooks/02_preprocessing_experiments.ipynb`
  - 다양한 전처리 조합 실험
  - 파라미터 그리드 서치
  - 시각화 및 비교 분석
  - 최적 파라미터 도출

### 검증 기준
✅ 노이즈가 많은 샘플에서 OCR 정확도 향상 확인
✅ 전처리 전/후 이미지 품질 비교 시각화
✅ 최적 파라미터 세트 문서화
✅ 처리 시간이 이미지당 1초 이내 유지

---

## Phase 4: 통합 파이프라인 및 배치 처리 (2-3일)

### 목표
전체 워크플로우를 통합하고 여러 이미지를 효율적으로 처리하는 시스템을 완성합니다.

### 작업 항목

#### 4.1 후처리 모듈 구현
- [ ] `src/postprocessing/text_cleaner.py`
  ```python
  import re

  class TextCleaner:
      def clean(self, text):
          # 특수문자 제거 (의료 기호 제외)
          text = re.sub(r'[^\w\s\.\:\±\%]', '', text)

          # 연속 공백 정규화
          text = re.sub(r'\s+', ' ', text)

          # 앞뒤 공백 제거
          return text.strip()

      def fix_common_errors(self, text):
          # OCR 오류 패턴 수정
          replacements = {
              'O': '0',  # 문맥에 따라
              'l': '1',
              'S': '5'
          }
          # 스마트 교체 로직
          return text
  ```

- [ ] `src/postprocessing/validator.py` - 결과 검증
  ```python
  class ResultValidator:
      def validate_date(self, text):
          """날짜 형식 검증"""
          date_patterns = [
              r'\d{4}-\d{2}-\d{2}',
              r'\d{4}/\d{2}/\d{2}'
          ]
          # 검증 로직

      def validate_measurement(self, text):
          """측정값 범위 검증"""
          # GA: 4-42 weeks
          # BPD: 20-100 mm
          # FL: 10-80 mm
          pass
  ```

#### 4.2 통합 파이프라인
- [ ] `src/pipeline/ocr_pipeline.py`
  ```python
  class OCRPipeline:
      def __init__(self, config_path):
          self.config = Config(config_path)
          self.preprocessor = PreprocessingPipeline(self.config)
          self.ocr_engine = EasyOCREngine(self.config)
          self.postprocessor = TextCleaner()
          self.validator = ResultValidator()

      def process(self, image_path):
          # 1. 이미지 로드
          image = cv2.imread(image_path)

          # 2. 전처리
          preprocessed = self.preprocessor.process(image)

          # 3. OCR
          raw_result = self.ocr_engine.recognize(preprocessed)

          # 4. 후처리
          cleaned_text = self.postprocessor.clean(raw_result['text'])

          # 5. 검증
          is_valid = self.validator.validate(cleaned_text)

          return {
              'text': cleaned_text,
              'confidence': raw_result['confidence'],
              'valid': is_valid,
              'raw_result': raw_result
          }
  ```

#### 4.3 배치 처리 스크립트
- [ ] `scripts/batch_process.py`
  ```python
  import glob
  from pathlib import Path
  from tqdm import tqdm

  def batch_process(input_dir, output_dir):
      pipeline = OCRPipeline('configs/default.yaml')

      image_files = glob.glob(f"{input_dir}/**/*.png", recursive=True)

      for image_path in tqdm(image_files):
          result = pipeline.process(image_path)

          # 결과 저장
          output_path = Path(output_dir) / f"{Path(image_path).stem}.json"
          save_result(result, output_path)
  ```

#### 4.4 에러 핸들링
- [ ] 파일 읽기 오류 처리
- [ ] OCR 실패 시 재시도 로직
- [ ] 부분 실패 시 로그 기록 및 계속 진행
- [ ] 최종 요약 리포트 생성

#### 4.5 성능 메트릭 수집
- [ ] `src/utils/metrics.py` 확장
  ```python
  class PerformanceMetrics:
      def __init__(self):
          self.processing_times = []
          self.confidences = []
          self.success_count = 0
          self.failure_count = 0

      def record(self, processing_time, confidence, success):
          self.processing_times.append(processing_time)
          self.confidences.append(confidence)
          if success:
              self.success_count += 1
          else:
              self.failure_count += 1

      def summary(self):
          return {
              'avg_time': np.mean(self.processing_times),
              'avg_confidence': np.mean(self.confidences),
              'success_rate': self.success_count / (self.success_count + self.failure_count)
          }
  ```

### 검증 기준
✅ 10개 이상의 샘플 이미지 배치 처리 성공
✅ 에러 발생 시 적절한 로그 기록 및 복구
✅ 처리 완료 후 요약 리포트 생성
✅ 평균 OCR 신뢰도 0.75 이상 달성

---

## 테스트 전략

### 단위 테스트 (Phase 2-4 동안 병행)

#### 전처리 테스트
- [ ] `tests/test_preprocessing/test_enhancer.py`
  - CLAHE 적용 결과 검증
  - Gamma 보정 범위 확인
  - 출력 이미지 형식 검증

- [ ] `tests/test_preprocessing/test_noise_reducer.py`
  - Bilateral Filter 엣지 보존 확인
  - NLM 노이즈 제거 효과 검증

- [ ] `tests/test_preprocessing/test_binarizer.py`
  - Otsu vs Adaptive 비교
  - 이진화 임계값 검증

#### OCR 테스트
- [ ] `tests/test_ocr/test_easyocr.py`
  - 모델 로드 성공 확인
  - 알려진 텍스트 이미지 인식 검증
  - 신뢰도 임계값 테스트

#### 후처리 테스트
- [ ] `tests/test_postprocessing/test_text_cleaner.py`
  - 특수문자 제거 규칙 검증
  - 공백 정규화 확인

- [ ] `tests/test_postprocessing/test_validator.py`
  - 날짜 형식 검증 테스트
  - 측정값 범위 검증

#### 통합 테스트
- [ ] `tests/test_integration/test_pipeline.py`
  - 전체 파이프라인 end-to-end 테스트
  - 다양한 품질의 이미지로 테스트
  - 성능 벤치마크

### 테스트 커버리지 목표
- Phase 2: 60% 이상
- Phase 3: 70% 이상
- Phase 4: 80% 이상

---

## 데이터 준비

### 샘플 이미지 수집
- [ ] `data/sample_images/raw/` - 원본 초음파 이미지 5-10장
  - 다양한 품질 (고품질, 중품질, 저품질)
  - 다양한 측정값 포함
  - 다양한 날짜 형식

- [ ] `data/sample_images/annotated/` - 정답 레이블
  - 각 이미지의 올바른 텍스트 기록
  - JSON 형식으로 저장

### 의료 용어 데이터베이스
- [ ] `data/medical_terms/abbreviations.json`
  ```json
  {
    "GA": "Gestational Age",
    "BPD": "Biparietal Diameter",
    "FL": "Femur Length",
    "AC": "Abdominal Circumference",
    "HC": "Head Circumference",
    "EFW": "Estimated Fetal Weight"
  }
  ```

---

## 문서화

### 코드 문서화
- [ ] 모든 public 함수/클래스에 docstring 작성
- [ ] 타입 힌트 추가 (Python 3.9+ 형식)
- [ ] 복잡한 알고리즘에 주석 추가

### 프로젝트 문서
- [ ] `claudedocs/architecture.md` - 아키텍처 설계 문서
- [ ] `claudedocs/preprocessing_guide.md` - 전처리 가이드
- [ ] `claudedocs/performance_report.md` - 성능 분석 리포트

### API 문서
- [ ] 각 모듈의 사용 예시 작성
- [ ] 설정 파일 옵션 상세 설명

---

## 품질 관리

### 코드 스타일
- [ ] Black으로 자동 포맷팅 적용
  ```bash
  black src/ tests/ scripts/
  ```

- [ ] Flake8으로 린팅 검사
  ```bash
  flake8 src/ tests/ --max-line-length=100
  ```

- [ ] MyPy로 타입 체크
  ```bash
  mypy src/ --ignore-missing-imports
  ```

### Git 워크플로우
- [ ] Feature 브랜치 전략 사용
  - `feature/preprocessing`
  - `feature/ocr-integration`
  - `feature/pipeline`

- [ ] 커밋 메시지 규칙
  ```
  feat: Add CLAHE image enhancement
  fix: Fix confidence calculation bug
  docs: Update preprocessing guide
  test: Add binarization unit tests
  ```

---

## 마일스톤 체크리스트

### MVP 완성 (v0.1)
- [ ] ✅ 프로젝트 구조 완성
- [ ] 단일 이미지 OCR 정상 동작
- [ ] 기본 전처리 적용
- [ ] JSON 결과 출력
- [ ] 기본 테스트 작성

### 고급 기능 (v0.2)
- [ ] 고급 전처리 알고리즘
- [ ] 배치 처리 기능
- [ ] 성능 메트릭 수집
- [ ] 통합 테스트 80%+

### 의료 특화 (v0.3)
- [ ] 의료 용어 사전 통합
- [ ] 컨텍스트 검증
- [ ] Tesseract 앙상블
- [ ] 구조화된 출력

---

## 트러블슈팅 가이드

### 예상 문제 및 해결책

#### 1. EasyOCR 모델 다운로드 실패
**증상**: 첫 실행 시 모델 다운로드 오류
**해결**: 수동으로 모델 다운로드 또는 프록시 설정

#### 2. GPU 메모리 부족
**증상**: CUDA out of memory 에러
**해결**: CPU 모드로 전환 또는 배치 크기 감소

#### 3. 낮은 OCR 정확도
**증상**: 신뢰도 < 0.5
**해결**: 전처리 파라미터 조정, 이미지 품질 확인

#### 4. 한글 인식 실패
**증상**: 한글 텍스트 인식 안 됨
**해결**: EasyOCR 언어 설정에 'ko' 추가 확인

---

## 다음 단계 (v1.0을 위한 준비)

### API 서버 개발
- FastAPI 프레임워크 사용
- RESTful API 엔드포인트 설계
- Swagger 문서 자동 생성

### 웹 인터페이스
- React 또는 Vue.js 프론트엔드
- 이미지 업로드 UI
- 결과 시각화 대시보드

### 배포 준비
- Docker 컨테이너화
- CI/CD 파이프라인 구축
- 클라우드 배포 (AWS/GCP)

---

## 참고 자료

### 논문 및 문서
- EasyOCR 공식 문서: https://github.com/JaidedAI/EasyOCR
- OpenCV 전처리 가이드: https://docs.opencv.org/
- 의료 영상 전처리 베스트 프랙티스

### 유사 프로젝트
- Medical Image OCR 사례 연구
- DICOM 이미지 처리 프로젝트

### 커뮤니티
- Stack Overflow - OCR 태그
- Reddit r/computervision
- OpenCV 포럼
