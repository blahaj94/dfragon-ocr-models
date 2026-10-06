# DFRAGON OCR 모델

DFRAGON용 파인튜닝 인식 모델 한 개와 대응 문자 사전을 관리합니다. 버전은 Git 커밋으로 구분하며, 모델별 폴더나 학습 코드는 두지 않습니다.

현재 등록한 모델은 **한국어 PP-OCRv5 기반 8에폭 FP16 ONNX**입니다. ONNX Runtime CUDA에서 검증한 파일을 변환 없이 보관합니다. **현재 Desktop의 WASM, 가변 너비 계약을 충족하는 모델은 아닙니다.** 부모 앱에서 바로 사용할 수 있는 상태로 간주하지 않습니다.

## 파일 배치

```text
model.onnx       # 가중치를 포함한 단일 FP16 ONNX 인식 모델
characters.txt   # 같은 모델의 문자 사전, 순서 유지
README.md
LICENSE          # 기반 모델과 구현의 Apache-2.0 라이선스 원문
NOTICE           # 출처와 변경 내역
.gitattributes
.gitignore
```

모델과 사전은 루트에 함께 등록하고 같은 커밋으로 관리합니다. `.gitattributes`는 두 파일의 원본 바이트를 유지합니다. 일반 Git 파일을 사용하며 Git LFS나 별도 external data 파일은 필요하지 않습니다.

## 등록 모델의 형식

| 항목 | 값 |
| --- | --- |
| 기반 모델 | `korean_PP-OCRv5_mobile_rec`, 합성 닉네임으로 파인튜닝 |
| 체크포인트 | 10에폭 학습 후 Val CER로 선택한 8에폭 |
| 저장 및 연산 정밀도 | FP16, 부동소수점 initializer 489개가 FLOAT16 |
| 입력 | `images`, float32 BGR NCHW **`[1,3,48,320]` 고정** |
| 출력 | 첫 출력 `fetch_name_0`, float32 CTC **`[1,40,9263]`** |
| ONNX | IR 8, 기본 opset 17 |
| 파일 크기 | 13,236,332 bytes, 약 12.62 MiB |
| 문자 사전 | 9,261자, UTF-8, BOM 없음, 한 줄에 한 문자 |
| 검증한 실행 경로 | ONNX Runtime 1.23.2, CUDA Execution Provider, RTX 3080 Ti |

입출력 텐서는 float32를 유지하지만 모델 내부의 FP16 가중치와 연산이 FP32로 바뀌는 것은 아닙니다. 입력 너비를 임의로 바꾸거나 batch를 늘려 사용할 수 없습니다. ONNX 입력 선언만 가변으로 바꾸는 방식도 검증하지 않았습니다.

`characters.txt`에는 중복, 빈 행, 공백 문자가 없습니다. **문자의 순서를 바꾸면 안 됩니다.** CTC blank는 0번이고, 사전의 각 문자를 이어 붙인 뒤 공백을 마지막 클래스로 추가합니다. 따라서 출력 클래스 수는 `9,261 + 2 = 9,263`입니다. 각 시점의 최대 확률 클래스를 선택하고 연속 중복과 blank를 제거하는 기존 CTC 디코딩을 사용합니다.

### 검증에 사용한 전처리

이 모델은 다음 조건의 이미지를 사용해 학습하고 평가했습니다.

1. 닉네임 영역을 그레이스케일로 디코딩합니다.
2. Otsu 반전 이진화(`THRESH_BINARY_INV | THRESH_OTSU`)로 흰 배경, 검은 글자 형태를 만듭니다. 가우시안 블러는 사용하지 않습니다.
3. 이진화 이미지를 BGR 3채널로 디코딩합니다.
4. 높이를 48로 맞추고 너비는 `min(320, ceil(48 × 원본너비 / 원본높이))`로 리사이즈합니다.
5. float32 CHW로 변환한 뒤 `(pixel / 255 - 0.5) / 0.5`로 정규화합니다. 오른쪽을 정규화 텐서의 0으로 채워 `[3,48,320]`으로 맞춥니다.
6. batch 축을 추가해 `[1,3,48,320]`으로 입력합니다.

리사이즈와 정규화는 고정한 PaddleOCR revision의 [RecResizeImg 구현](https://github.com/PaddlePaddle/PaddleOCR/blob/b03f46425e8ff4442b268ce449e3eef758146cd4/ppocr/data/imaug/rec_img_aug.py)을 따랐습니다. Desktop의 기존 반전 회색조 입력만으로 아래 평가 수치를 재현했다고 간주하지 않습니다.

별도 평가에서는 이진화 후 텍스트 좌우 경계에서 각각 원본 픽셀 5px의 여유를 남기고, 위아래 원본 픽셀 행을 각각 1줄 제거했습니다. 좌우 경계는 높이 `max(3, ceil(이미지 높이 × 0.25))` 이상, 면적 4 이상인 검은 연결 요소에서 찾습니다. 면적 2 이상, 텍스트 세로 범위와 겹치고 간격이 `max(2, ceil(이미지 높이 × 0.18))` 이하인 인접 조각도 포함합니다. 좌우 여유는 원본 이미지 경계에서 제한하고, 기준 요소가 없으면 전체 폭을 유지합니다. 경계 계산에 정답 문자열은 사용하지 않습니다. 이 크롭은 모델 내부 기능이 아니라 외부 전처리입니다.

## 평가 결과

실제 게임 이미지 Test 478장, 두 입력 조건에서 총 956건을 비교했습니다. ONNX FP32와 FP16의 **예측 문자열이 956건 모두 동일**했습니다. 미지원 문자 6장은 원본 평가에 남기고, 지원 문자 472장 결과를 별도로 집계했습니다.

| 입력 조건 | 전체 478장 일치율 | 전체 CER | 지원 문자 472장 일치율 | 지원 문자 CER |
| --- | ---: | ---: | ---: | ---: |
| 이진화, 크롭 없음 | 450/478, 94.14% | 1.84% | 450/472, 95.34% | 1.58% |
| 이진화, 상하 1px 제거, 좌우 5px 여유 | 457/478, 95.61% | 1.46% | 457/472, 96.82% | 1.20% |

같은 Test를 반복 사용했고 크롭 규칙은 실패 사례를 보고 조정했습니다. 새로운 미사용 평가 세트의 일반화 성능을 의미하지 않습니다. Confidence는 닉네임 전체가 맞을 확률로 보정된 값이 아닙니다.

RTX 3080 Ti, batch 1에서 FP16의 평균 `session.run` 시간은 약 6.60ms였습니다. 전처리와 CTC 디코딩은 제외했고, 측정 중 학습 계산은 정지했지만 학습 모델의 GPU 메모리는 유지했습니다. 다른 장치나 Desktop WASM에서 같은 속도를 보장하지 않습니다.

## 출처와 라이선스

- 기반 모델: PaddleOCR 팀의 [korean_PP-OCRv5_mobile_rec](https://huggingface.co/PaddlePaddle/korean_PP-OCRv5_mobile_rec), 모델 카드의 라이선스는 Apache-2.0입니다.
- 기반 구현: [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), revision `b03f46425e8ff4442b268ce449e3eef758146cd4`.
- 변경 내역: 합성 Train 닉네임 555,225개로 파인튜닝, 문자 사전 필터링 및 확장, Val 기준 8에폭 선택, ONNX 내보내기, I/O float32를 유지한 FP16 변환.
- 도구: 학습 Paddle GPU 3.2.2, 내보내기 Paddle CPU 3.1.1 및 Paddle2ONNX 2.0.2rc3, FP16 변환 onnxconverter-common 1.16.0.
- 라이선스 원문과 고지는 [LICENSE](LICENSE), [NOTICE](NOTICE)에 있습니다. 학습 이미지, 원본 닉네임 목록, 글꼴 파일, 추론 런타임은 배포하지 않습니다.

| 대상 | SHA-256 |
| --- | --- |
| `model.onnx` | `a6dc80ae798fb7f5e8e21fccf948ae057188e26a6a958a5964cbe7b261144f3c` |
| `characters.txt` | `7d0db05340a8ad68618fb0645a31420455d8ae2adc7537913b573c71115fd85b` |
| 원본 8에폭 체크포인트 | `69c28e17173491e89251b1f8e8b0cbcc1ac56be5e4e2cef88dc5bf51df09bc00` |
| 공식 초기 가중치 | `8975dede5e0c2f47e0a7712b3d79ffdc766972f872fd0441ebcccd9d77cd52a3` |

## Desktop 호환 조건과 연동

| 항목 | 기존 Desktop 목표 계약 | 현재 등록 모델 |
| --- | --- | --- |
| 실행 | ONNX Runtime Web WASM | CUDA에서 검증, WASM 실행 검증 없음 |
| 입력 너비 | `[1,3,48,W]`, 가변 너비 | `[1,3,48,320]`, 고정 너비 |
| 입력과 출력 dtype | float32 | float32, 내부는 FP16 |
| 일반 캡처 전처리 | 반전 회색조 | Otsu 반전 이진화로 학습 및 평가 |

[ONNX Runtime의 FP16 문서](https://onnxruntime.ai/docs/performance/model-optimizations/float16.html)는 CPU 실행의 FP16 연산 제약을 설명합니다. I/O가 float32라는 이유만으로 내부 FP16 연산을 사용하는 이 모델이 WASM에서 동작한다고 가정하지 않습니다. 현재 파일은 검증한 FP16 모델의 보관용이며, Desktop 실행 경로에 맞추는 작업은 별도입니다.

부모 저장소 [dfragon](https://github.com/blahaj94/dfragon)은 이 저장소를 `apps/desktop/models/finetuned` submodule로 가져옵니다. 부모는 특정 커밋을 고정하므로 모델을 올리는 것만으로 기존 앱이나 빌드가 바뀌지 않습니다. 이번 등록에서 부모 submodule이나 모델 선택 설정을 변경하지 않았습니다.

**아래 절차는 Desktop 실행 호환성을 별도로 맞추고 검증한 모델을 준비한 이후에 사용합니다.** 현재 FP16 파일을 WASM 경로에 바로 연결하는 절차가 아닙니다.

```sh
git submodule update --init --recursive apps/desktop/models/finetuned
git -C apps/desktop/models/finetuned fetch origin
git -C apps/desktop/models/finetuned checkout <호환성을-검증한-모델-커밋-SHA>
```

부모의 작업 브랜치에서 `apps/desktop/ocr-model.config.mjs` 마지막 줄을 `export default fineTunedOcrModel`로 바꾸고 검증합니다.

```sh
pnpm --filter @dfragon/desktop prepare:ocr-assets
pnpm --filter @dfragon/desktop build
git add apps/desktop/models/finetuned apps/desktop/ocr-model.config.mjs
```

부모 변경은 별도 PR로 전달합니다. 빌드의 파일 및 입출력 검사와 별도로 실제 실행 provider, 사전 순서, 입력 전처리와 정확도를 검증해야 합니다. 현재 등록 작업에서는 Desktop 빌드와 WASM 추론을 실행하지 않았습니다.
