# SMOTE Tutorial

`SMOTE.py` 스크립트와 데모 데이터 `sample.csv`는 원작자(dnl8145@gmail.com)가 배포한 Docker Image에 포함된 파일입니다.

Docker Image에는 python3와 pandas 패키지가 설치되어 있습니다.

아래 단락은 원작자의 Docker Image에 포함된 `readme.rtf` 파일의 내용입니다.

> SMOTE.py execute with arguments which are absolute path of the csv file and critical value.
> if GB(Gender Bias) is larger than the input argument Critical, the smaller gender gets upsampled.
> It uses only default python library and pandas. And I already installed the python and the pandas in this images.
> If you have any question please mail me to dnl8145@gmail.com

## Docker Image

원작자의 Docker Image는 Docker Hub에 `dnl8145/smote_dslab`이라는 이름으로 등록되어있습니다.

### Download

docker가 사용 가능한 환경에서 아래 명령어를 이용해 다운로드 할 수 있습니다.

```bash
$ docker pull dnl8145/smote_dslab:0.1
```

### 이미지 내부로 진입하기

튜토리얼 파일은 `/tmp/smote/` 디렉토리에 위치해 있습니다.

```bash
$ docker run -itd --name smote_tutorial dnl8145/smote_dslab:0.1
$ docker exec -it smote_tutorial /bin/bash
```

### 실행

이미지 내부에서 아래 명령어를 통해 스크립트를 실행할 수 있습니다.

```bash
$ python smote.py sample.csv 0.5
```

## 시각화 보고서

`SMOTE_Tutorial.ipynb` 파일은 `SMOTE.py`에 구현된 알고리즘을 데이터에 적용해보고 그 과정을 시각화한 보고서입니다.

Jupyter Notebook 형식으로 작성되어 있습니다.
