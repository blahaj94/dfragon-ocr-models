# DFRAGON OCR 모델

DFRAGON Desktop에 포함할 파인튜닝 인식 모델 한 개와 해당 문자 사전을 관리합니다. 버전은 Git 커밋으로 구분하며, 모델별 폴더나 학습 코드는 두지 않습니다.

## 파일 배치

```text
model.onnx       # 사용자가 추가할 ONNX 인식 모델
characters.txt   # 같은 모델을 학습할 때 사용한 문자 사전
README.md
.gitattributes
.gitignore
```

현재는 저장소 구조만 준비한 상태이며 실제 모델과 사전은 아직 없습니다. 두 파일을 루트에 함께 추가하고 같은 커밋으로 관리합니다. 빈 ONNX 파일이나 임시 사전을 실제 모델 대신 넣지 않습니다. 모델의 출처와 필요한 라이선스 고지는 실제 모델을 등록할 때 함께 기록합니다.

## Desktop 호환 형식

- `model.onnx`: 가중치를 포함한 단일 ONNX 파일입니다. ONNX Runtime Web WASM에서 실행하며 float32 BGR `[1, 3, 48, W]`, 높이 48과 가변 너비를 사용합니다. 값은 `[-1, 1]`로 정규화합니다. 첫 출력은 float32 CTC `[1, steps, classes]`입니다.
- `characters.txt`: UTF-8, BOM 없이 한 줄에 한 문자를 기록합니다. 중복, 빈 행, 공백 문자는 제외합니다. CRLF와 마지막 개행 한 개는 허용합니다. **학습 때 사용한 순서를 그대로 유지합니다.**
- CTC blank는 0번, 공백은 마지막 클래스로 앱이 추가합니다. 모델 출력 클래스 수는 사전 문자 수 + 2입니다. 문자 확장 모델을 올릴 때는 확장한 ONNX와 사전을 함께 교체합니다.
- 일반 캡처는 반전 회색조 전처리를 사용합니다. Paddle 학습 체크포인트나 별도 external data 파일을 요구하는 ONNX는 직접 사용할 수 없습니다.

`.gitattributes`는 ONNX와 사전의 원본 바이트를 유지합니다. 이 저장소는 일반 Git 파일을 사용하며 Git LFS 설정은 포함하지 않습니다.

## DFRAGON에 반영하기

부모 저장소 [dfragon](https://github.com/blahaj94/dfragon)은 이 저장소를 `apps/desktop/models/finetuned` submodule로 가져옵니다. 부모는 특정 커밋을 고정하므로 여기에 모델을 올리는 것만으로 기존 앱이나 빌드가 바뀌지는 않습니다.

부모 저장소의 작업 브랜치에서 실행합니다.

```sh
git submodule update --init --recursive apps/desktop/models/finetuned
git -C apps/desktop/models/finetuned fetch origin
git -C apps/desktop/models/finetuned checkout <모델을-포함한-커밋-SHA>
```

그다음 부모의 `apps/desktop/ocr-model.config.mjs` 마지막 줄을 `export default fineTunedOcrModel`로 바꾸고 검증합니다.

```sh
pnpm --filter @dfragon/desktop prepare:ocr-assets
pnpm --filter @dfragon/desktop build
git add apps/desktop/models/finetuned apps/desktop/ocr-model.config.mjs
```

부모의 변경을 PR로 전달합니다. 빌드 시 파일과 입출력, 사전 클래스 수를 검사하며 실패하면 기본 모델로 대체하지 않습니다. 사전의 의미상 순서와 실제 정확도는 개발자 평가의 정답 이미지로 별도 확인해야 합니다.
