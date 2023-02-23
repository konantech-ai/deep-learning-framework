# __KONAN AI setup manual__

## __개발 환경__
 * Microsoft Visual Studio Professional 2019 (16.11)
 * C++14 (ISO/IEC 14882)
 * CUDA 11.3
 * cuDNN 8.2.1
 * cpprestsdk 2.10.18
 * Python 3.9.5
<br><br>

## __C/C++ 엔진 빌드시 필요한 라이브러리__
 * NVIDIA CUDA Toolkit 11.3.x : <https://developer.nvidia.com/cuda-toolkit-archive>
 * NVIDIA cuDNN 8.2.1 for CUDA 11.x : <https://developer.nvidia.com/rdp/cudnn-archive>
 * cpprestsdk 2.10.18 : <https://github.com/microsoft/cpprestsdk/>
 * OpenCV 4.5.2 : <https://sourceforge.net/projects/opencvlibrary/files/4.5.2/opencv-4.5.2-vc14_vc15.exe/download>
<br><br>

## __라이브러리 설치 방법__
### 1. CUDA, cuDNN 설치 
      > NVIDIA 그래픽 드라이버 설치 ( 이미 설치되어 있는 경우 Skip, 빠른설치 X, 지포스 익스피리언스 제외 )  

      > CUDA 11.3.x 설치 ( 이미 설치되어 있는 경우 Skip, 빠른설치 X, 지포스 익스피리언스 제외 )  
      >> 다운로드 링크 : https://developer.nvidia.com/cuda-toolkit-archive  

      > cuDNN 8.2.1 압축파일 다운로드  
      >> 다운로드 링크 : https://developer.nvidia.com/rdp/cudnn-archive  
      >> 압축 해제  
      >> bin, include, lib 폴더 복사  
      >> C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3 폴더 내부에 붙여넣기  
### 2. vcpkg 설치를 위한 git bash 설치 
      > 다운로드 링크 : https://git-scm.com/download/win  
      > 64-bit Git for Windows Setup 다운로드  
      > 기본 체크 사항으로 Next 및 Install  
### 3. cpprestsdk 설치
      > cpprestsdk 설치를 위한 vcpkg 설치  
      >> git bash 실행  
      >> cd c:  
      >> git clone https://github.com/microsoft/vcpkg  
      >> ./vcpkg/bootstrap-vcpkg.bat  

      > cpprestsdk 설치(소요시간 5~10분)  
      >> git bash 실행  
      >> cd c:  
      >> cd vcpkg  
      >> ./vcpkg install cpprestsdk cpprestsdk:x64-windows  
      >> ./vcpkg integrate install  
      >> 컴퓨터 재시작  
<br><br>

## __가상 환경 설정 ( Windows - Anaconda )__
### 1. Miniconda3 설치 
      > 다운로드 링크 : https://docs.conda.io/en/latest/miniconda.html  
      > Miniconda3 Windows 64-bit 다운로드  
      > 기본으로 체크 사항으로 Next 및 Install  
### 2. 가상 환경 생성 및 활성화, 사용자 환경 변수 등록 
      > 관리자 모드로 Anaconda Prompt 실행
      >> conda create -n <ENV_NAME> python=3.9.5
      >> conda activate <ENV_NAME>
      >> setx KONANAI_PYTHON_PATH %CONDA_PREFIX%

      > 정상적으로 경로가 등록되었는지 확인을 위해 Anaconda Prompt 종료 -> 재실행
      >> conda activate <ENV_NAME>
      >> echo %KONANAI_PYTHON_PATH%
### 3. 연산 보조 및 시각화 패키지 설치
      > 관리자 모드로 Anaconda Prompt 실행
      >> conda activate <ENV_NAME>
      >> pip install numpy pandas matplotlib
### 4. CUDA, cuDNN 설치 (공용 라이브러리는 가상환경보다 conda의 기본환경에 설치를 권장함)
      > 관리자 모드로 Anaconda Prompt 실행
      > pip install이 아닌 conda install을 사용
      >> conda activate <ENV_NAME>
      >> conda install -c nvidia cudatoolkit=11.3
      >> conda install -c nvidia cudnn=8.2.1

      > 옵션 설명
      >> conda install -c : channel search (conda package)
### 5. PyTorch 설치 (공용 라이브러리는 가상환경보다 conda의 기본환경에 설치를 권장함)
      > conda install -c pytorch pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1
<br><br>

## __Git을 이용한 소스코드 다운로드 및 환경 변수 설정__ 
### 1. 소스코드를 다운로드 할 경로로 이동 후 디렉토리 생성
      > git bash 실행 
      >> cd <WORK_PATH>
      >> mkdir <SOLUTION_DIR_NAME>
### 2. 생성한 디렉토리로 이동 후 git clone 명령어를 통해 다운로드
      > cd <SOLUTION_DIR_NAME>
      > git clone https://github.com/konantech-ai/deep-learning-framework.git .
      > 옵션 설명
      >> git clone -b platform_ai_group --depth 1 https://github.com/konantech-ai/deep-learning-framework.git .
      >> git clone [-b <BRANCH_NAME>] [--depth <COMMIT_COUNT>] [<REMOTE_REPOSITORY_URL>] [.]  
      >> -b : 특정 브랜치만 다운로드 받음  
      >> --depth : 최신 N개 커밋만 다운로드 받음  
      >> <REMOTE_REPOSITORY_URL> : https://github.com/konantech-ai/deep-learning-framework.git  
      >> . : 하위 디렉토리를 생성하지 않고 현재 위치에 바로 다운로드
### 3. 컴파일 위치를 사용자 환경 변수로 등록
      > 시작 -> cmd (반드시 cmd 사용) -> 생성한 디렉토리로 이동  
      > setx KONANAI_PATH %cd%\  
### 4. C++, Python 디버깅을 위한 라이브러리 설치
      > 릴리즈만 사용하는 경우 해당 항목 스킵.
      > cmake 설정에 의해 Python의 디버그 라이브러리는 KONANAI_PYTHON_PATH 에서 찾도록 되어있음.
      > 가상 환경으로 설치한 Python은 디버그 라이브러리를 지원하지 않아서 디버그 모드에서 빌드 시 문제가 발생함.
      > Local Python의 디버그 라이브러리를 활용해야 함.

      > Python 3.9.5 다운로드 : https://www.python.org/downloads/release/python-395/
      > Windows installer (64-bit) 다운로드
      > 기본 항목으로 설치.
      > 설치 완료 -> 시작 -> 앱 -> Python 3.9.5 -> 수정
      > Python installer 실행됨 -> Modify -> Next
      >> Download debugging symbols 체크.
      >> Download debug binary ( requires VS2017 or later ) 체크.
      >> install

      > 관리자 모드로 Anaconda Prompt 실행.
      >> cd C:/Users/%username%/AppData/Local/Programs/Python/Python39
      >> setx KONANAI_PYTHON_PATH %cd%
<br><br>

## __Visual Studio 2019 환경 설정__ 
     > Visual Studio 2019의 기본 설정에 의해 CMakeLists.txt 파일로 연결된 모든 프로젝트는 폴더에 캐시를 자동 생성함  
     > 캐시를 자동 생성하지 않도록 옵션 변경
     >> Visual Studio 2019 상단 탭 > Tools > Options > CMake > General > When cache is out of date 항목  
     >> Never run configure step automatically 로 변경  

     > Python 개발 환경 설치
     >> 시작 -> visual studio installer -> 수정 -> 'Python 개발' 설치
     >> visual sutudio 2019 상단 탭 -> 디버그 -> 옵션 -> Python -> 디버깅 -> Python 표준 라이브러리 디버깅 사용 체크
<br><br>

## __Python 개발용 konanai 패키지 설치 ( Windows - Anaconda )__
### 1. conda 가상 환경 활성화
      > conda activate <ENV_NAME>
### 2. Python 패키지 소스코드가 위치한 경로로 이동
      > cd %KONANAI_PATH%\source\python
### 3-1. 패키지 설치 (일반 사용자용)
      > pip install .
   * 파일 복사 형태로 설치 : 엔진 소스 코드가 변경되어도 패키지는 유지됨
### 3-2. 패키지 설치 (엔진 개발자용)
      > pip install -e .
   * 파일 링크 형태로 설치 : 엔진 소스 코드를 수정하면 패키지 내용이 변경됨
<br><br>
