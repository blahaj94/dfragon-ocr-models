# DFRAGON OCR 모델

DFRAGON용 파인튜닝 인식 모델 한 개와 대응 문자 사전을 관리합니다. 버전은 Git 커밋으로 구분하며, 모델별 폴더나 학습 코드는 두지 않습니다.

현재 등록한 모델은 **한국어 PP-OCRv5 기반 8에폭 FP32 ONNX**입니다. Desktop의 ONNX Runtime Web WASM 입력 계약에 맞춰 **가변 너비 `[1,3,48,W]`**로 내보냈습니다. 원본 FP32 체크포인트를 사용했고 문자 사전은 유지했습니다.

이번 변경에서는 파일 구조와 입출력 형식만 확인했습니다. **WASM 실행, Desktop 빌드, 인식률 재평가는 요청 범위에서 제외했습니다.** 아래의 과거 평가 수치는 이번 가변 너비 파일의 실행 검증 결과가 아닙니다.

## 파일 배치

```text
model.onnx       # 가중치를 포함한 단일 FP32, 가변 너비 ONNX 인식 모델
characters.txt   # 같은 모델의 문자 사전, 순서 유지
README.md
LICENSE          # 기반 모델과 구현의 Apache-2.0 라이선스 원문
NOTICE           # 출처와 변경 내역
.gitattributes
.gitignore
```

모델과 사전은 루트에 함께 관리합니다. `.gitattributes`는 원본 바이트를 유지합니다. 일반 Git 파일을 사용하며 Git LFS나 별도 external data 파일은 필요하지 않습니다. 모델을 교체하더라도 반드시 같은 체크포인트에 대응하는 사전을 사용합니다.

## 등록 모델의 형식

| 항목 | 값 |
| --- | --- |
| 기반 모델 | `korean_PP-OCRv5_mobile_rec`, 합성 닉네임으로 파인튜닝 |
| 체크포인트 | 10에폭 학습 후 Val CER로 선택한 8에폭 |
| 정밀도 | FP32, 저장된 부동소수점 텐서 923개가 FLOAT, FP16 텐서 없음 |
| 입력 | `images`, float32 BGR NCHW **`[1,3,48,W]`**, batch 1과 높이 48 고정 |
| 출력 | 첫 출력 `fetch_name_0`, float32 CTC **`[1,steps,9263]`** |
| ONNX | IR 8, opset 17, 표준 ONNX 연산만 사용 |
| 파일 크기 | 26,794,558 bytes, 약 25.55 MiB |
| 문자 사전 | 9,261자, UTF-8, BOM 없음, 한 줄에 한 문자 |
| 대상 실행 경로 | ONNX Runtime Web WASM, 이번 파일의 실제 실행은 미확인 |

입력 너비는 추적 단계부터 가변으로 설정해 내보냈습니다. 고정 너비 모델의 입력 선언만 바꾼 파일이 아닙니다. 너비에 따른 출력 길이 `steps`도 가변입니다. 모델 파일에는 가중치가 포함되어 있고 CUDA 전용 또는 별도 사용자 정의 연산을 넣지 않았습니다.

`characters.txt`에는 중복, 빈 행, 공백 문자가 없습니다. **문자의 순서를 바꾸면 안 됩니다.** CTC blank는 0번이고, 사전의 각 문자를 이어 붙인 뒤 공백을 마지막 클래스로 추가합니다. 따라서 출력 클래스 수는 `9,261 + 2 = 9,263`입니다. 각 시점의 최대 확률 클래스를 선택하고 연속 중복과 blank를 제거하는 기존 CTC 디코딩을 사용합니다.

### 입력 전처리와 기존 평가 조건

모델 입력은 BGR, float32 CHW이며 값은 `[-1,1]`로 정규화합니다. batch 축을 추가한 실제 입력 너비가 `W`가 됩니다. Desktop의 기존 일반 캡처는 반전 회색조를 사용하지만, 이 체크포인트의 학습과 아래 과거 평가는 **Otsu 반전 이진화**를 사용했습니다. 전처리 차이에 따른 결과는 이번에 다시 평가하지 않았습니다.

과거 고정 너비 평가 조건은 다음과 같습니다.

1. 닉네임 영역을 그레이스케일로 디코딩합니다.
2. Otsu 반전 이진화(`THRESH_BINARY_INV | THRESH_OTSU`)로 흰 배경, 검은 글자 형태를 만듭니다. 가우시안 블러는 사용하지 않습니다.
3. 이진화 이미지를 BGR 3채널로 디코딩합니다.
4. 높이를 48로 맞추고 너비는 `min(320, ceil(48 × 원본너비 / 원본높이))`로 리사이즈합니다.
5. float32 CHW로 변환한 뒤 `(pixel / 255 - 0.5) / 0.5`로 정규화합니다. 오른쪽을 정규화 텐서의 0으로 채워 `[3,48,320]`으로 맞춥니다.
6. batch 축을 추가해 `[1,3,48,320]`으로 입력합니다.

리사이즈와 정규화는 고정한 PaddleOCR revision의 [RecResizeImg 구현](https://github.com/PaddlePaddle/PaddleOCR/blob/b03f46425e8ff4442b268ce449e3eef758146cd4/ppocr/data/imaug/rec_img_aug.py)을 따랐습니다. 현재 모델의 가변 너비 형식과 과거에 평가한 너비 320 조건을 구분합니다.

별도 평가에서는 이진화 이미지에서 텍스트 좌우 경계를 구하고 각각 원본 픽셀 5px의 여유를 남긴 뒤, 위아래 원본 픽셀 행을 각각 1줄 제거했습니다. 좌우 경계는 높이 `max(3, ceil(이미지 높이 × 0.25))` 이상, 면적 4 이상인 검은 연결 요소에서 찾습니다. 면적 2 이상, 텍스트 세로 범위와 겹치고 간격이 `max(2, ceil(이미지 높이 × 0.18))` 이하인 인접 조각도 포함합니다. 좌우 여유는 원본 이미지 경계에서 제한하고, 기준 요소가 없으면 전체 폭을 유지합니다. 경계 계산에 정답 문자열은 사용하지 않습니다. 이 크롭은 모델 내부 기능이 아니라 외부 전처리입니다.

## 참고: 같은 체크포인트의 과거 평가

아래 수치는 **이전 고정 너비 FP32 및 FP16 모델**의 결과입니다. 실제 게임 이미지 Test 478장, 두 입력 조건에서 총 956건을 비교했고 두 정밀도의 예측 문자열이 모두 동일했습니다. 미지원 문자 6장은 원본 평가에 남기고, 지원 문자 472장 결과를 별도로 집계했습니다.

| 입력 조건 | 전체 478장 일치율 | 전체 CER | 지원 문자 472장 일치율 | 지원 문자 CER |
| --- | ---: | ---: | ---: | ---: |
| 이진화, 크롭 없음 | 450/478, 94.14% | 1.84% | 450/472, 95.34% | 1.58% |
| 이진화, 상하 1px 제거, 좌우 5px 여유 | 457/478, 95.61% | 1.46% | 457/472, 96.82% | 1.20% |

이번 가변 너비 FP32 모델로 위 수치가 재현되는지는 확인하지 않았습니다. 같은 Test를 반복 사용했고 크롭 규칙은 실패 사례를 보고 조정했으므로 새로운 미사용 평가 세트의 일반화 성능도 의미하지 않습니다. Confidence는 닉네임 전체가 맞을 확률로 보정된 값이 아닙니다.

이전 고정 너비 모델의 native ONNX Runtime 평균 `session.run` 시간은 i5-10400 CPU 2스레드의 FP32 약 65.93ms, RTX 3080 Ti의 FP32 약 6.75ms, FP16 약 6.60ms였습니다. batch 1, 전처리와 CTC 디코딩 제외 조건입니다. **현재 가변 너비 파일의 WASM 속도는 측정하지 않았습니다.**

## 출처와 라이선스

- 기반 모델: PaddleOCR 팀의 [korean_PP-OCRv5_mobile_rec](https://huggingface.co/PaddlePaddle/korean_PP-OCRv5_mobile_rec), 모델 카드의 라이선스는 Apache-2.0입니다.
- 기반 구현: [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), revision `b03f46425e8ff4442b268ce449e3eef758146cd4`.
- 변경 내역: 합성 Train 닉네임 555,225개로 파인튜닝, 문자 사전 필터링 및 확장, Val 기준 8에폭 선택, 원본 FP32 가중치에서 가변 너비 ONNX로 내보내기.
- 도구: 학습 Paddle GPU 3.2.2, 내보내기 Paddle CPU 3.1.1 및 Paddle2ONNX 2.0.2rc3. 이번 파일에는 FP16 변환이나 플랫폼 전용 ONNX Runtime 그래프 최적화를 적용하지 않았습니다.
- 라이선스 원문과 고지는 [LICENSE](LICENSE), [NOTICE](NOTICE)에 있습니다. 학습 이미지, 원본 닉네임 목록, 글꼴 파일, 추론 런타임은 배포하지 않습니다.

| 대상 | SHA-256 |
| --- | --- |
| `model.onnx` | `76c0cfcc4dd9d12ea6657083700d2df0a1e7e664b99520ef6a7da25405ff81e6` |
| `characters.txt` | `7d0db05340a8ad68618fb0645a31420455d8ae2adc7537913b573c71115fd85b` |
| 원본 8에폭 체크포인트 | `69c28e17173491e89251b1f8e8b0cbcc1ac56be5e4e2cef88dc5bf51df09bc00` |
| 공식 초기 가중치 | `8975dede5e0c2f47e0a7712b3d79ffdc766972f872fd0441ebcccd9d77cd52a3` |

## Desktop 연결

모델 파일 형식은 기존 Desktop의 float32 BGR `[1,3,48,W]` 입력과 float32 CTC 출력 계약에 맞췄습니다. 구조 확인 범위는 가변 너비, 입출력 dtype, 사전과 출력 클래스 수, 표준 연산 domain, external data 참조 부재와 파일 해시입니다. **WASM 추론과 Desktop 빌드는 수행하지 않았으며 성공으로 보고하지 않습니다.**

부모 저장소 [dfragon](https://github.com/blahaj94/dfragon)은 이 저장소를 `apps/desktop/models/finetuned` submodule로 가져옵니다. 부모는 특정 커밋을 고정하므로 모델을 올리는 것만으로 기존 앱이나 빌드가 바뀌지 않습니다. 이번 변경에서 부모 submodule이나 모델 선택 설정은 변경하지 않았습니다.

부모의 작업 브랜치에서 모델 커밋을 선택하는 절차입니다.

```sh
git submodule update --init --recursive apps/desktop/models/finetuned
git -C apps/desktop/models/finetuned fetch origin
git -C apps/desktop/models/finetuned checkout <모델을-포함한-커밋-SHA>
```

부모의 `apps/desktop/ocr-model.config.mjs` 마지막 줄을 `export default fineTunedOcrModel`로 바꾸고 부모 변경을 별도 PR로 전달합니다. 다음은 부모 연결 시 사용할 기존 검사 명령이며 이번 작업에서는 실행하지 않았습니다.

```sh
pnpm --filter @dfragon/desktop prepare:ocr-assets
pnpm --filter @dfragon/desktop build
git add apps/desktop/models/finetuned apps/desktop/ocr-model.config.mjs
```
