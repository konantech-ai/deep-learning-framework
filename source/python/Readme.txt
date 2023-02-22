################################################################################

[ README ]

kai -> kai_torchtest -> src -> api_python -> sample
  convert_sample
    unit test file
  reference
    tut_ver01 : konanai_cpp.exe로 결과 비교 가능
    tut_ver02 : konanai_cpp.exe로 결과 비교 불가능
  나머지 폴더
    model sample : konanai_cpp.exe로 결과 비교 가능

kai -> work -> cache -> tconn_python
  해당 폴더는 반드시 존재해야 함 : work 폴더 부터 없으면 만들도록 수정 필요해보임

################################################################################

[ 작업 유의사항 ]

KONANAI_PATH 환경변수 필요
  - C:\ 사용자 지정 경로 \kai2021\

Visual Studio 2019
  - CUDA
    CUDA 11.3은 Visual Studio 2022 컴파일러 지원 안함
  
  - x64-Release Mode 사용
    x64-Debug Mode로 빌드 시 Python 패키지 중 디버그 모드 지원 안하는 경우 종종 있음

  - CMake 설정
    Tools -> Options -> CMake -> General -> When cache is out of date
    Never run configure step automatically 로 변경

  - 탭, 들여쓰기 사이즈
    도구 -> 옵션 -> 텍스트 편집기 -> C/C++ -> 탭
    탭 크기 : 4
    들여쓰기 크기 : 4

Python 3.9
  - KONANAI_PYTHON_PATH 환경변수 필요
    로컬의 경우
      C:\Users\사용자 이름\AppData\Local\Programs\Python\Python39
    가상 환경의 경우
      가상환경의 Python PATH
  - 디버그 라이브러리 사용
    윈도우 -> 앱 및 기능 -> Python -> 수정 -> Modify -> Optional Feature -> Next -> Advanced Options
    Download debugging symbols
    Download debug binaries (requires VS 2017 or later)

  - Python Package Install
    kai -> konanai -> src
    setup.py
    requirements.txt
    콘솔로 이동 ( cmd, git bash, etc ... )
    conda 사용자는 가상환경 활성화 후 실행
    - 정식설치 (파일 복사) : 'pip install .'
    - 개발자모드 설치 (파일 링크 연결) : 'pip install -e .'

RESTfulAPI
  - cpp SDK 설치 필요
    [ Installation : Environment ]
    vcpkg  : https://github.com/Microsoft/vcpkg
    https://github.com/microsoft/vcpkg/blob/master/README_ko_KR.md
    git bash
    cd c: && git clone https://github.com/microsoft/vcpkg
    ./vcpkg/bootstrap-vcpkg.bat
    [ Installation : cpprsetsdk ****2.10.18**** ]
    https://github.com/microsoft/cpprestsdk/blob/master/README.md
    API Document : https://microsoft.github.io/cpprestsdk/namespaces.html
    Sample Code : https://github.com/Microsoft/cpprestsdk/wiki/Samples
    cd c: && cd vcpkg
    ./vcpkg install cpprestsdk cpprestsdk:x64-windows
    ./vcpkg integrate install
  - sample code
    [ example ]
      kai -> konanai -> src -> api_python -> sample -> restful
      restful_server_sample.py 실행 : (사내 네트워크의 데스크톱 1 에서 실행 -> 데스크톱 1의 Addr 입력)
      restful_client_sample.py 실행 : (사내 네트워크의 데스크톱 2 에서 실행 -> 데스크톱 1의 Addr 입력)
    [ server sample code ]
      RESTfulAPI* TestServer;
      TestServer = new RESTfulAPI("http://xxx.xxx.xxx.xxx:xxxx", true, false);
      delete TestServer;
    [ client sample code ]
      RESTfulAPI* TestClient;
      TestClient = new RESTfulAPI("http://xxx.xxx.xxx.xxx:xxxx", false, true);
      TestClient->Sample_ClientAsyncRequestGET();
      TestClient->Sample_ClientAsyncRequestPUT();
      delete TestClient;

vcpkg
  - 명령어 설명
    x32 패키지 설치 : ./vcpkg install [package name]
    x64 패키지 설치 : ./vcpkg install [package name]:x64-windows
    Example : x32 x64 패키지 동시 설치 : ./vcpkg install [package name] [package name]:x64-windows

kai2021-> kai-> CMakeLists.txt
  - 가상환경 사용하지 않는 경우 아래 내용 주석처리
    file(COPY "C:/ProgramData/Miniconda3/envs/AI-DF/python39.dll"
    DESTINATION  ${CMAKE_BINARY_DIR}/..)
    file(COPY "C:/ProgramData/Miniconda3/envs/AI-DF/python39_d.dll"
    DESTINATION  ${CMAKE_BINARY_DIR}/..)

################################################################################

Python sample file 활용을 위해 필요한 packages
  1. struct (as st)
  2. numpy (as np)
  3. os, sys
  4. time
  5. math
  6. pathlib의 Path

활용 시 필요한 부분
  - Tonumpy_Mnist_train( ), Tonumpy_Mnist_test( ) 함수에서 각각 image, label의 array를 받아오기 위해서
    최상단에서 data_types를, Tonumpy_Mnist_train( ) 함수 내에서 nDim, dataType, dataFormat, dataSize를 global로 선언 필요
  - path 정보 활용을 위해 아래 코드 작성 (현재 버전의 코드에서는 사용하지 않았지만 추후 cache 설정 시 사용 예정)
	curdir = os.path.dirname(__file__)
	pardir = Path(curdir).parent.parent
	abspardir = os.path.abspath(pardir)
	sys.path.append(abspardir)

################################################################################