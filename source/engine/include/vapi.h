#pragma once

#ifdef V_EXPORTS
#ifdef VAPI
#undef VAPI
#endif
#define VAPI __declspec(dllexport)
#else
#define VAPI __declspec(dllimport)
#endif

#ifdef FOR_LINUX
#ifdef VAPI
#undef VAPI
#endif
#define VAPI __attribute__((__visibility__("default")))
#endif

#include "verrors.h"
#include "vtypes.h"
#include "vvalue.h"
#include "vwrapper.h"


/// Defines an alias representing buffer for result data/
typedef void VCbForwardModule(VHSession hSession, const VExBuf* pCbInstBuf, const VExBuf* pCbStatusBuf, const VExBuf* pCbTensorBuf, const VExBuf** ppResultBuf);
typedef void VCbBackwardModule(VHSession hSession, const VExBuf* pCbInstBuf, const VExBuf* pCbStatusBuf, const VExBuf* pCbParamBuf, const VExBuf* pCbGradBuf, const VExBuf** ppResultBuf);
typedef void VCbClose(VHSession hSession, const VExBuf* pResultBuf);


/// Defines an alias representing buffer for result data
typedef void VCbForwardFunction(VHSession hSession, void* pHandlerAux, VHFunction hFunction, int nInst, const VExBuf* pTensorListBuf, const VExBuf* pArgDictBuf, const VExBuf** ppResultBuf);
typedef void VCbBackwardFunction(VHSession hSession, void* pHandlerAux, VHFunction hFunction, int nInst, const VExBuf* pGradListBuf, const VExBuf* pTensorListBuf, const VExBuf* pArgDictBuf, int nth, const VExBuf** ppResultBuf);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 프레임워크 엔진 기능을 이용하는 데 사용할 세션 연결을 만들고 세션 핸들을 반환합니다.
///
/// @param [out]	phSession	세션 핸들 값
/// @param [in]	   	pDictBuf 	VDict 형식 옵션 정보의 래핑 구조체 주소
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Session_open(VHSession * phSession, const VExBuf * pDictBuf);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 세션 연결을 해제합니다.
///
/// @param [in]	hSession	연결을 해제할 세션 핸들
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Session_close(VHSession hSession);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 세션 연결된 프레임워크 엔진의 버전정보를 반환합니다.
///
/// @param [in]		hSession 	세션 핸들
/// @param [out]	psVersion	프레임워크 엔진의 버전 문자열
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Session_getVersion(VHSession hSession, string * psVersion);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션에서 난수값 발생에 이용할 씨앗값을 지정합니다.
///
/// @param [in]		hSession 	세션 핸들
/// @param [in]		rand_seed	난수 씨앗값, 고정된 정수값 혹은 time(NULL) 등을 이용할 수 있음
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Session_seedRandom(VHSession hSession, int64 rand_seed);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션에서 이용 가능한 쿠다 디바이스의 갯수를 반환합니다.
///
/// @param [in]		hSession	 	세션 핸들
/// @param [out]	pnDeviceCount	쿠다 디바이스의 갯수
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Session_getCudaDeviceCount(VHSession hSession, int* pnDeviceCount);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션의 딥러닝 처리 과정에서 역전파 처리를 위한 기울기 정보 관리 해제 여부를 지정합니다.
///
/// @param [in]		hSession	세션 핸들
/// @param [in]		no_grad 	True면 추론의 효율적 수행을 위해 역전파와 기울기에 대한 고려가 없는 방식으로 동작
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Session_setNoGrad(VHSession hSession, bool no_grad);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션의 연산추적기 사용 해제 여부를 지정합니다.
///
/// @param [in]		hSession	세션 핸들
/// @param [in]		no_tracer 	True로 지정하면 연산추적기를 사용하지 않게 되지만 이 경우 속도와 안정성이 저하될 수 있습니다.
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Session_setNoTracer(VHSession hSession, bool no_tracer);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션을 통해 이용 가능한 긱종 빌트인 성분들의 명칭을 얻어옵니다.
///
/// @param [in]		hSession	세션 핸들
/// @param [out]	ppDictBuf	VDict 형식 반환값의 래핑 구조체 주소, 래핑을 풀면 사용가능한 기본레이어, 복합레이어, 손실함수, 옵티마이저, 연산함수, 내장모델 등의 목록을 얻을 수 있다.
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Session_getBuiltinNames(VHSession hSession, const VExBuf** ppDictBuf);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션에서 지정된 이름의 모듈을 처리하기 위해 설정한 계산그래프 구성용 수식 내용을 반환합니다.
///
/// @param [in]		hSession	세션 핸들
/// @param [in]		sBuiltin   	수식을 얻어올 내장 모듈명으로서  단순 레이어나 복합 레이어의 명칭
/// @param [out]	psFormula	데이터 식별번호 리스트 정보의 래핑 구조체 포인터
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Session_getFormula(VHSession hSession, string sBuiltin, string* psFormula);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션에 모듈 구조를 매크로로 등록합니다. 등록된 매크로는 Module_createMacro() 호출을 통해 새로운 모듈 생성에 이용할 수 있습니다.
///
/// @param [in]		hSession 	세션 핸들
/// @param [in]		macroName	매크로 이름
/// @param [in]		hModule  	매크로 내용으로 등록할 구조를 갖는 모듈 객체
/// @param [in]		pDictBuf 	VDict 형식 옵션 정보의 래핑 구조체 주소
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Session_registMacro(VHSession hSession, string macroName, VHModule hModule, const VExBuf * pDictBuf);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션에서 커스텀 모듈의 실행을 위해 호출할 커스텀 모듈 실행용 콜백 함수를 등록합니다.
///
/// @param [in]	   	hSession	세션 핸들
/// @param [in]		pFunc   	등록할 커스텀 모듈 실행용 콜백 함수 포인터
/// @param [in]		pInst   	호출시 콜백 함수의 첫번 째 인자로 사용할 인스턴스 식별용 포인터 정보
/// @param [in]		pAux		호출시 콜백 함수의 두번 째 인자로 사용할 보조 정보를 위한 포인터 정보
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Session_registCustomModuleExecFunc(VHSession hSession, VCbCustomModuleExec * pFunc, void* pInst, void* pAux);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션에서 커스텀 모듈 실행 중 제공한 버퍼 공간의 해제를 요청할 때 호출할 콜백 함수를 등록합니다.
///
/// @param [in]	   	hSession	세션 핸들
/// @param [in]		pFunc   	등록할 버퍼 공간 해제 요청용 콜백 함수 포인터
/// @param [in]		pInst   	호출시 콜백 함수의 첫번 째 인자로 사용할 인스턴스 식별용 포인터 정보
/// @param [in]		pAux		호출시 콜백 함수의 두번 째 인자로 사용할 보조 정보를 위한 포인터 정보
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Session_registFreeReportBufferFunc(VHSession hSession, VCbFreeReportBuffer * pFunc, void* pInst, void* pAux);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션에서 순전파 처리 중 호출할 콜백 함수를 콜백 호출에 대한 각종 설정 정보와 함께 등록합니다.
///
/// @param [in]	   	hSession  	세션 핸들
/// @param [in]		pCbFunc   	순전파 처리 과정 추적 및 제어를 위헤 엔진이 호출할 콜백 함수 포인터
/// @param [in]		pCbClose  	콜백 함수 수행 후 뒷정리 작업을 위해 엔진이 호출할 콜백 함수 포인터
/// @param [in]	   	pFilterBuf	호출 시점 지정을 위한 필터 정보 래핑 구조체 주소
/// @param [in]	   	pCbInstBuf	호출 주체 지정을 위한 필터 정보 래핑 구조체 주소
/// @param [out]	pnId	  	콜백 지정에 대한 등록 번호를 받아올 변수의 포인터
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Session_addForwardCallbackHandler(VHSession hSession, VCbForwardModule * pCbFunc, VCbClose * pCbClose, const VExBuf * pFilterBuf, const VExBuf * pCbInstBuf, int* pnId);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션에서 역전파 처리 중 호출할 콜백 함수를 콜백 호출에 대한 각종 설정 정보와 함께 등록합니다.
///
/// @param [in]	   	hSession  	세션 핸들
/// @param [in]		pCbFunc   	순전파 처리 과정 추적 및 제어를 위헤 엔진이 호출할 콜백 함수 포인터
/// @param [in]		pCbClose  	콜백 함수 수행 후 뒷정리 작업을 위해 엔진이 호출할 콜백 함수 포인터
/// @param [in]	   	pFilterBuf	호출 시점 지정을 위한 필터 정보 래핑 구조체 주소
/// @param [in]	   	pCbInstBuf	호출 주체 지정을 위한 필터 정보 래핑 구조체 주소
/// @param [out]	pnId	  	콜백 지정에 대한 등록 번호를 받아올 변수의 포인터
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Session_addBackwardCallbackHandler(VHSession hSession, VCbBackwardModule* pCbFunc, VCbClose * pCbClose, const VExBuf * pFilterBuf, const VExBuf * pCbInstBuf, int* pnId);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션에 등록했던 순전파 혹은 역전파 처리 중 호출 콜백 함수 지정을 철회하여 관련 정보를 삭제합니다.
///
/// @param 	[in]	hSession	세션 핸들
/// @param 	[in]	nId			콜백 지정시에 받아왔던 등록 번호
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Session_removeCallbackHandler(VHSession hSession, int nId);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션이 사용자 정의 함수를 처리할 때 호출할 콜백 함수들을 등록합니다.
///
/// @param  [in]	hSession		세션 핸들
/// @param  [in]	pCbAux		 	함수 호출시 인자로 콜백 수진측에 전달할 보조 정보 포인터
/// @param  [in]	pFuncForward 	사용자 정의 함수의 순전파 처리를 수행시키기 위해 호출할 콜백 함수 포인터
/// @param  [in]	pFuncBackward	사용자 정의 함수의 역전파 처리를 수행시키기 위해 호출할 콜백 함수 포인터
/// @param  [in]	pCbClose	 	사용자 정의 함수 처리를 종료한 후 마무리 작업을 수행시키기 위해 호출할 콜백 함수 포인터
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Session_setFuncCbHandler(VHSession hSession, void* pCbAux, VCbForwardFunction* pFuncForward, VCbBackwardFunction* pFuncBackward, VCbClose * pCbClose);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션에서 제공한 래핑 구조체를 참조 활용 후 사용 해제하여 메모리 반납을 가능하게 합니다.
///
/// @param [in]		hSession	세션 핸들
/// @param [in]		pBuf		다른 API 함수에서 const VExBuf** 형식의 인자를 통해 서버가 보내주었던 래핑 구조체 버퍼의 주소
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Session_freeExchangeBuffer(VHSession hSession, const VExBuf * pBuf);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션에서 가장 최근에 발생했던 오류의 오류 코드 값을 반환합니다.
///
/// @param [in]		hSession	세션 핸들
/// @param [out]	pRetCode	최근 발생한 오류의 오류 코드를 받아올 정수형 변수 포인터
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Session_getLastErrorCode(VHSession hSession, VRetCode * pRetCode);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션에서 가장 최근에 발생했던 오류의 오류 메시지 값을 반환합니다.
///
/// @param [in]		hSession		세션 핸들
/// @param [out]	ppErrMessages	최근 발생한 오류의 오류 메시지를 받아올 문자열 변수 포인터
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Session_getLastErrorMessageList(VHSession hSession, const VExBuf** ppErrMessages);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션에서 지정된 핸들값에 해당하는 객체를 찾아 그 코어에 할당된 고유번호를 반환합니다.
/// 반환되는 값은 보통 디버깅 용도로 이용합니다.
///
/// @param [in]		hSession		세션 핸들
/// @param [in]		handle			고유 번호 조회를 원하는 객체의 핸들
/// @param [out]	pnId			조회된 고유번호를 저장할 정수형 변수의 주소, 없으면 -1
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Session_getIdForHandle(VHSession hSession, VHandle handle, int* pnId);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션의 자원 관리 모니터링을 위해 삭제되지 않은 객체의 목록 및 관련 정보를 반환합니다.
///
/// @param  [in]	hSession			세션 핸들
/// @param  [in]	sessionOnly			true면 연결 세션에 대해서만 조사, false면 세션에 관계 없이 모든 객체 조사
/// @param 	[out]	ppLsBuf				삭제되지 않은 객체의 목록 및 관련 정보를 포함하는 사전 정보의 래핑 구조체를 받아롤 포인터 주소
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Session_getLeakInfo(VHSession hSession, bool sessionOnly, const VExBuf * *ppLsBuf);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션에서 새로운 모듈 객체를 생성하도록 하고 생성된 객체의 핸들을 반환합니다.
///
/// @param [in]		hSession	세션 핸들
/// @param [out]	phModule	생성된 객체의 핸들을 받아올 변수 포인터
/// @param [in]		sBuiltin	V_Session_getBuiltinNames()에서 알려주는 단순 레이어 유형 혹은 복합 레이어 유형 이름 중 하나로서 생성할 모듈의 종류 지정
/// @param [out]	psName  	엔진이 생성된 모듈의 식별을 위해 부여한 이름 문자열을 받아올 변수 포인터
/// @param [in]		pDictBuf 	VDict 형식 옵션 정보의 래핑 구조체 주소
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Module_create(VHSession hSession, VHModule * phModule, string sBuiltin, string * psName, const VExBuf * pDictBuf);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션에서 등록된 매크로 정보를 이용해 새로운 모듈 객체를 생성하도록 하고 생성된 객체의 핸들을 반환합니다.
///
/// @param [in]		hSession	세션 핸들
/// @param [out]	phModule	생성된 객체의 핸들을 받아올 변수 포인터
/// @param [in]		sMacroName	V_Session_registMacro() 호출로 등록했던 모듈 생성에 이용할 매크로 정보의 이름
/// @param [out]	psName  	엔진이 생성된 모듈의 식별을 위해 부여한 이름 문자열을 받아올 변수 포인터
/// @param [in]		pDictBuf 	VDict 형식 옵션 정보의 래핑 구조체 주소, 매크로 인자가 전달되는 통로
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Module_createMacro(VHSession hSession, VHModule * phModule, string sMacroName, string * psName, const VExBuf * pDictBuf);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션에서 지정된 수식을 이용해 사용자 정의 모듈 객체를 생성하도록 하고 생성된 객체의 핸들을 반환합니다.
///
/// @param [in]		hSession	세션 핸들
/// @param [out]	phModule	생성된 객체의 핸들을 받아올 변수 포인터
/// @param [in]		sName	 	사용자 정의 모듈에 부여할 모듈 이름
/// @param [in]		sFormula	사용자 정의 모듈 객체 생성에 이용할 수식 문자열
/// @param [in]	   	pParamBuf	생성된 모듈 객체가 가질 학습용 파라미터 구성을 알려줄 VDict 형식 설정 정보의 래핑 구조체 주소
/// @param [in]		pDictBuf 	VDict 형식 옵션 정보의 래핑 구조체 주소
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Module_createUserDefinedLayer(VHSession hSession, VHModule* phModule, string sName, string sFormula, const VExBuf* pParamBuf, const VExBuf * pDictBuf);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션에서 모듈 구성 정보로 지정된 내용의 모듈 객체를 생성하도록 하고 생성된 객체의 핸들을 반환합니다.
///
/// @param [in]		hSession	세션 핸들
/// @param [out]	phModule	생성된 객체의 핸들을 받아올 변수 포인터
/// @param [in]		pDictBuf 	VDict 형식 모듈 구성 정보의 래핑 구조체 주소
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Module_load(VHSession hSession, VHModule* phModule, const VExBuf* pDictBuf);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션에서 지정된 복합 레이어 모듈에 자식 모듈을 막내로 추가합니다.
///
/// @param [in]		hSession		세션 핸들
/// @param [in]		hModule			막내를 추가할 부모 모듈의 핸들.
/// @param [in]		hChildModule	막내로 추가될 자식 모듈의 핸들.
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Module_appendChildModule(VHSession hSession, VHModule hModule, VHModule hChildModule);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션에서 지정된 모듈에 단일 텐서를 입력으로 주어 순전파 처리 결과를 계산합니다.
///
/// @param [in]		hSession	세션 핸들
/// @param [in]		hModule		순전파 처리를 수행할 모듈의 핸들.
/// @param [in]	   	train   	학습 모드에면 true, 추론 모드이면 false
/// @param [in]		hInput  	순전파 처리 과정의 입력으로 사용할 텐서 핸들
/// @param [out]	phOutput	순전파 처리 과정의 출력으로 생성된 텐서 핸들을 받아올 핸들 변수 포인터
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Module_evaluate(VHSession hSession, VHModule hModule, bool train, VHTensor hInput, VHTensor * phOutput);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션에서 지정된 모듈에 텐서 목록을 입력으로 주어 순전파 처리 결과를 계산합니다.
///
/// @param [in]		hSession	세션 핸들
/// @param [in]		hModule		순전파 처리를 수행할 모듈의 핸들.
/// @param [in]	   	train   	학습 모드에면 true, 추론 모드이면 false
/// @param [in]		pXsBuf  	순전파 처리 과정의 입력으로 사용할 텐서 핸들 목록의 래핑 구조체 주소
/// @param [out]	ppYsBuf		순전파 처리 과정의 출력으로 생성된 텐서 핸들 목록을 받아올 래핑 구조체 변수 포인터
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Module_evaluateEx(VHSession hSession, VHModule hModule, bool train, const VExBuf * pXsBuf, const VExBuf * *ppYsBuf);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션에서 지정된 모듈이 갖고 있는 파라미터/기울기 텐서들을 모아 구성된 파라미터군 객체의 핸들을 반환합니다.
///
/// @param [in]		hSession		세션 핸들
/// @param [in]		hModule			파라미터 정보를 가져올 모듈의 핸들.
/// @param [out]	phParameters	구성된 파라미터 객체의 핸들을 받아올 핸들 변수 포인터
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Module_getParameters(VHSession hSession, VHModule hModule, VHParameters * phParameters);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션의 한 모듈이 갖고 있는 자식 모듈들을 다른 모듈에 복사합니다.
///
/// @param [in]		hSession		세션 핸들
/// @param [in]		hModule			자식 노드들을 복사할 대상 모듈의 핸들.
/// @param [in]		hSrcModule		자식 노드 정보를 제공할 소스 모듈의 핸들
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Module_copyChildren(VHSession hSession, VHModule hModule, VHModule hSrcModule);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션의 한 모듈에 대하여 이름이 일치하는 자식 모듈을 찾아 그 핸들을 반환합니다.
///
/// @param [in]		hSession		세션 핸들
/// @param [in]		hModule			자식 노드를 찾을 부모 모듈의 핸들.
/// @param [in] 	name		 	자식 모듈의 이름, 생성시 부여된 이름이며 속성 확인을 통해서도 알 수 있음
/// @param [in]	   	bChildOnly   	자식 노드에 한정할지 여부, false이면 후손 노드 전체를 탐색함
/// @param [out]	phChildModule	발견된 자식 모듈의 핸들을 받아올 핸들 변수 포인터
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Module_fetchChild(VHSession hSession, VHModule hModule, string name, bool bChildOnly, VHModule* phChildModule);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션의 한 모듈에 대하여 자식 모듈 전체의 핸들을 리스트 형식으로 반환합니다.
///
/// @param [in]		hSession		세션 핸들
/// @param [in]		hModule			자식 노드를 찾을 부모 모듈의 핸들.
/// @param [out]	ppBuf   		자식 노드들의 핸들 리스트를 받아올 래핑 구조체 변수 포인터
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Module_getChildrenModules(VHSession hSession, VHModule hModule, const VExBuf** ppBuf);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션에서 지정된 모듈을 템플릿으로 삼아 특정 입력 형상을 처리하도록 튜닝된 새로운 모듈 객체를 만들고 만들어진 객체의 핸들을 반환합니다.
///
/// @param [in]		hSession			세션 핸들
/// @param [in]		hModule				템플릿 역할을 할 기존 모듈 객체의 핸들.
/// @param [in]   	pShapeBuf			처리를 원하는 입력 형상 내용을 저장한 래핑 구조체 포인터
/// @param [in]	   	pDictBuf			모듈 객체 생성 과정에 반영할 부수적 정보를 VDict 형태로 저장한 래핑 구조체 포인터
/// @param [out]	phExpandedModule	새로 생성된 모듈 객체의 핸들을 받아올 핸들 변수 포인터
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Module_expand(VHSession hSession, VHModule hModule, const VExBuf* pShapeBuf, const VExBuf* pDictBuf, VHModule * phExpandedModule);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션에서 지정된 모듈을 템플릿으로 삼아 특정 디바이스에서 처리를 수행할 새로운 모듈 객체를 만들고 만들어진 객체의 핸들을 반환합니다. 단 기존 모듈이 이미 해당 디바이스를 지원하면 기존 모듈 객체의 핸들을 반환합니다.
///
/// @param [in]		hSession		세션 핸들
/// @param [in]		hModule			템플릿 역할을 할 기존 모듈 객체의 핸들.
/// @param [in]	   	device		  	처리를 수행할 디바이스 이름, 'gpu', 'cuda', 'cuda:0' 등의 값
/// @param [out]	phDeviceModule	기존 모듈 객체 혹은 새로 생성된 모듈 객체의 핸들을 받아올 핸들 변수 포인터
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Module_toDevice(VHSession hSession, VHModule hModule, string device, VHModule * phDeviceModule);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션의 지정된 모듈에 대한 제반 정보를 사전 형식으로 반환합니다.
///
/// @param [in]		hSession	세션 핸들
/// @param [in]		hModule		정보를 찾을 대상 모듈 객체의 핸들.
/// @param [out]	ppDictBuf	모듈 객체의 제반 정보를 담은 사전 형식 정보를 받아올 래핑 구조체 변수 포인터
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Module_getModuleInfo(VHSession hSession, VHModule hModule, const VExBuf** ppDictBuf);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션의 지정된 모듈을 루트로 하는 모듈 트리에 대해 모듈들이 갖는 파라미터 값을 일괄 설정합니다.
///
/// @param [in]		hSession	세션 핸들
/// @param [in]		hModule		파라미터 내용을 업로드할 대상 모듈 트리의 루트 모듈 객체의 핸들.
/// @param [in]		pTensorBuf	업로드할 모듈 파라미터들의 내용이 담신 사전 형식 정보의 래핑 구조체 포인터
/// @param [in]		mode		업로드할 모듈 파라미터 정보의 구성 형식을 알려주는 업로드 모드 지정자
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Module_setParamater(VHSession hSession, VHModule hModule, const VExBuf * pTensorBuf, string mode);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션에서 지정된 모듈이 순전파 처리 중 호출할 콜백 함수를 콜백 호출에 대한 각종 설정 정보와 함께 등록합니다.
///
/// @param [in]	   	hSession  	세션 핸들
/// @param [in]		hModule   	콜백 함수를 호출할 모듈 핸들
/// @param [in]		pCbFunc   	순전파 처리 과정 추적 및 제어를 위헤 엔진이 호출할 콜백 함수 포인터
/// @param [in]		pCbClose  	콜백 함수 수행 후 뒷정리 작업을 위해 엔진이 호출할 콜백 함수 포인터
/// @param [in]	   	pFilterBuf	호출 시점 지정을 위한 필터 정보 래핑 구조체 주소
/// @param [in]	   	pCbInstBuf	호출 주체 지정을 위한 필터 정보 래핑 구조체 주소
/// @param [out]	pnId	  	콜백 지정에 대한 등록 번호를 받아올 변수의 포인터
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Module_addForwardCallbackHandler(VHSession hSession, VHModule hModule, VCbForwardModule* pCbFunc, VCbClose* pCbClose, const VExBuf * pFilterBuf, const VExBuf * pCbInstBuf, int* pnId);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션에서 지정된 모듈이 역전파 처리 중 호출할 콜백 함수를 콜백 호출에 대한 각종 설정 정보와 함께 등록합니다.
///
/// @param [in]	   	hSession  	세션 핸들
/// @param [in]		hModule   	콜백 함수를 호출할 모듈 핸들
/// @param [in]		pCbFunc   	순전파 처리 과정 추적 및 제어를 위헤 엔진이 호출할 콜백 함수 포인터
/// @param [in]		pCbClose  	콜백 함수 수행 후 뒷정리 작업을 위해 엔진이 호출할 콜백 함수 포인터
/// @param [in]	   	pFilterBuf	호출 시점 지정을 위한 필터 정보 래핑 구조체 주소
/// @param [in]	   	pCbInstBuf	호출 주체 지정을 위한 필터 정보 래핑 구조체 주소
/// @param [out]	pnId	  	콜백 지정에 대한 등록 번호를 받아올 변수의 포인터
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Module_addBackwardCallbackHandler(VHSession hSession, VHModule hModule, VCbBackwardModule* pCbFunc, VCbClose* pCbClose, const VExBuf * pFilterBuf, const VExBuf * pCbInstBuf, int* pnId);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션에서 지정된 모듈에 대해 등록했던 순전파 혹은 역전파 처리 중 호출 콜백 함수 지정을 철회하여 관련 정보를 삭제합니다.
///
/// @param [in]	hSession	세션 핸들
/// @param [in]	hModule   	콜백 함수를 삭제할 모듈 핸들
/// @param [in]	nId			콜백 지정시에 받아왔던 등록 번호
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Module_removeCallbackHandler(VHSession hSession, VHModule hModule, int nId);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션에서 지정된 모듈에 대해 현재 처리중인 데이터의 식별번호들을 알려줍니다. 이 값은 순전파/역전파 콜백 함수를 통해 유용하게 활용될 수 있습니다.
///
/// @param [in]	hSession	세션 핸들
/// @param [in]	hModule   	데이터 식별번호를 알려줄 대상 모듈의 핸들
/// @param [in]	pListBuf	데이터 식별번호 리스트 정보의 래핑 구조체 포인터
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Module_uploadDataIndex(VHSession hSession, VHModule hModule, const VExBuf* pListBuf);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션에서 지정된 모듈 핸들과 객체 간의 참조 관계를 해제하고 더 이상 필요가 없는 경우 객체를 삭제합니다.
///
/// @param [in]	hSession	세션 핸들
/// @param [in]	hModule   	엔진 내부 객체와의 참조 관계를 해제할 모듈 핸들
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Module_close(VHSession hSession, VHModule hModule);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션에서 새로운 텐서 객체를 생성하도록 하고 생성된 객체의 핸들을 반환합니다.
///
/// @param [in]		hSession	세션 핸들
/// @param [out]	phTensor	생성된 객체의 핸들을 받아올 변수 포인터
/// @param [in]		sBuiltin	생성할 텐서의 종류를 지정하지만 현재는 그 값을 이용하지 않고 무시합니다.
/// @param [in]		pDictBuf 	VDict 형식 옵션 정보의 래핑 구조체 주소
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Tensor_create(VHSession hSession, VHTensor * phTensor, string sBuiltin, const VExBuf* pDictBuf);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션에서 지정된 텐서 핸들과 객체 간의 참조 관계를 해제하고 더 이상 필요가 없는 경우 객체를 삭제합니다.
///
/// @param [in]	hSession	세션 핸들
/// @param [in]	hTensor   	엔진 내부 객체와의 참조 관계를 해제할 텐서 핸들
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Tensor_close(VHSession hSession, VHTensor hTensor);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션에서 지정된 텐서의 속성들을 지정합니다.
///
/// @param [in]	hSession	세션 핸들
/// @param [in]	hTensor   	속성을 지정할 텐서 핸들
/// @param [in]	pShapeBuf	텐서에 지정할 형상 정보를 담은 래핑 구조체 주소
/// @param [in]	dataType 	텐서에 지정할 데이터 타입
/// @param [in]	nDevice  	텐서에 지정할 디바이스 번호
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Tensor_setFeature(VHSession hSession, VHTensor hTensor, const VExBuf* pShapeBuf, VDataType dataType, int nDevice);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션에서 지정된 텐서의 속성들을 조회하여 반환합니다.
///
/// @param [in]		hSession	세션 핸들
/// @param [in]		hTensor   	속성을 조사할 텐서 핸들
/// @param [out]	ppShapeBuf	텐서의 형상 정보를 담은 래핑 구조체 주소를 저장할 변수의 포인터
/// @param [out]	pdataType 	텐서의 데이터 타입을 저장할 변수의 포인터
/// @param [out]	pnDevice  	텐서의 디바이스 번호를 저장할 변수의 포인터
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Tensor_getFeature(VHSession hSession, VHTensor hTensor, const VExBuf** ppShapeBuf, VDataType* pdataType, int* pnDevice);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션에서 지정된 텐서의 내용을 업로드하여 변경합니다.
///
/// @param [in]	hSession	세션 핸들
/// @param [in]	hTensor   	내용을 업로드할 텐서 핸들
/// @param [in]	pData		텐서에 지정할 데이터 내용을 담은 메모리 블록의 시작 주소
/// @param [in]	nByteSize 	업로드할 데이터 메모리 블록의 바이트 단위 크기
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Tensor_uploadData(VHSession hSession, VHTensor hTensor, void* pData, int64 nByteSize);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션에서 지정된 텐서의 내용을 반환하여 다운로드시켜 줍니다.
///
/// @param [in]	hSession	세션 핸들
/// @param [in]	hTensor   	내용을 다운로드할 텐서 핸들
/// @param [in]	pData		텐서 내용을 저장할 메모리 블록의 시작 주소
/// @param [in]	nByteSize 	다운로드할 데이터 메모리 블록의 바이트 단위 크기
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Tensor_downloadData(VHSession hSession, VHTensor hTensor, void* pData, int64 nByteSize);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션에서 지정된 텐서를 복사하되 디바이스가 변경된 새 텐서 객체를 만들고 그 핸들을 반환합니다.
///
/// @param [in]	 hSession	 세션 핸들
/// @param [out] phDevTensor 새로 생성된 텐서 객체의 핸들을 받아올 변수 포인터
/// @param [in]	 hTensor   	 복사할 텐서의 핸들
/// @param [in]	 nDevice  	 새로운 텐서에 지정할 디바이스 번호
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Tensor_toDevice(VHSession hSession, VHTensor *phDevTensor, VHTensor hTensor, int nDevice);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션에서 지정된 텐서를 손실함수 값으로 산출했던 순전파 과정에 대해 역전파 처리를 수행합니다.
///
/// @param [in]	 hSession	 세션 핸들
/// @param [in]	 hTensor   	 역전파를 시작할 텐서의 핸들, 손실함수 값으로 산출된 텐서이어야 함
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Tensor_backward(VHSession hSession, VHTensor hTensor);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션에서 지정된 텐서를 계산했던 순전파 과정에 대해 주어진 기울기를 이용한 역전파 처리를 수행합니다.
///
/// @param [in]	 hSession	 세션 핸들
/// @param [in]	 hTensor   	 역전파를 시작할 텐서의 핸들, 모듈이나 손실 객체의 순전파 과정에서 산출된 텐서이어야 함
/// @param [in]	 hGrad   	 역전파 시작 텐서에 대응시켜 사용할 기울기 정보를 갖는 텐서 핸들
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Tensor_backwardWithGradient(VHSession hSession, VHTensor hTensor, VHTensor hGrad);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션에서 새로운 손실 객체를 생성하도록 하고 생성된 객체의 핸들을 반환합니다.
///
/// @param [in]		hSession	세션 핸들
/// @param [out]	phLoss		생성된 객체의 핸들을 받아올 변수 포인터
/// @param [in]		sBuiltin	V_Session_getBuiltinNames()에서 알려주는 손실 유형유형 이름 중 하나로서 생성할 손실의 종류 지정
/// @param [in]		pDictBuf 	VDict 형식 옵션 정보의 래핑 구조체 주소
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Loss_create(VHSession hSession, VHLoss * phLoss, string sBuiltin, const VExBuf * pDictBuf);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션에서 지정된 손실 핸들과 객체 간의 참조 관계를 해제하고 더 이상 필요가 없는 경우 객체를 삭제합니다.
///
/// @param [in]	hSession	세션 핸들
/// @param [in]	hLoss   	엔진 내부 객체와의 참조 관계를 해제할 손실 핸들
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Loss_close(VHSession hSession, VHLoss hLoss);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션에서 지정된 손실 객체를 이용하여 손실 함수 값을 계산하고 그 결과를 반환합니다.
///
/// @param [in]		hSession		세션 핸들
/// @param [in]		hLoss   		손실 함수 값을 계산할 손실 객체 핸들
/// @param [in]		download_all	true이면 게산된 보조 항목 값들도 함께 반환, false이면 계산된 손실 함수 값만 반환
/// @param [in]		pPredsBuf   	손실함수 값 계산에 이용할 신경망 추정 텐서 핸들 목록의 래핑 구조체 주소
/// @param [in]		pYsBuf			손실함수 값 계산에 이용할 정답 텐서 핸들 목록의 래핑 구조체 주소
/// @param [out]	ppLsBuf			계산 결과 구해진 결과 텐서 핸들 목록을 받아올 래핑 구조체 변수의 주소
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Loss_evaluate(VHSession hSession, VHLoss hLoss, bool download_all, const VExBuf* pPredsBuf, const VExBuf* pYsBuf, const VExBuf * *ppLsBuf);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션에서 지정된 손실 객체를 이용하여 정확도 값을 계산하고 그 결과를 반환합니다.
///
/// @param [in]		hSession		세션 핸들
/// @param [in]		hLoss   		정확도 값을 계산할 손실 객체 핸들
/// @param [in]		download_all	true이면 게산된 보조 항목 값들도 함께 반환, false이면 계산된 정확도 값만 반환
/// @param [in]		pPredsBuf   	정확도 계산에 이용할 신경망 추정 텐서 핸들 목록의 래핑 구조체 주소
/// @param [in]		pYsBuf			정확도 계산에 이용할 정답 텐서 핸들 목록의 래핑 구조체 주소
/// @param [out]	ppAccBuf		계산 결과 구해진 결과 텐서 핸들 목록을 받아올 래핑 구조체 변수의 주소
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Loss_eval_accuracy(VHSession hSession, VHLoss hLoss, bool download_all, const VExBuf* pPredsBuf, const VExBuf* pYsBuf, const VExBuf* *ppAccBuf);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션에서 지정된 손실 객체에 저장된 손실함수 값 계산까지의 순전파 처리 과정을 활용하여 역전파를 수행합니다. 수행 결과는 각 모듈의 파라미터 기울기 정보 형태로 엔진 내부에 저장됩니다.
///
/// @param [in]		hSession		세션 핸들
/// @param [in]		hLoss   		손실 함수 계산에 이용되어 역전파 수행 정보를 갖고 있는 손실 객체 핸들
///
/// @returns	A VRetCode.
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Loss_backward(VHSession hSession, VHLoss hLoss);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션에서 새로운 평가 객체를 생성하도록 하고 생성된 객체의 핸들을 반환합니다.
///
/// @param [in]		hSession	세션 핸들
/// @param [out]	phMetric		생성된 객체의 핸들을 받아올 변수 포인터
/// @param [in]		sBuiltin	V_Session_getBuiltinNames()에서 알려주는 평가 유형 이름 중 하나로서 생성할 평가 객체의 종류 지정
/// @param [in]		pDictBuf 	VDict 형식 옵션 정보의 래핑 구조체 주소
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Metric_create(VHSession hSession, VHMetric * phMetric, string sBuiltin, const VExBuf* pDictBuf);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션에서 지정된 평가 핸들과 객체 간의 참조 관계를 해제하고 더 이상 필요가 없는 경우 객체를 삭제합니다.
///
/// @param [in]	hSession	세션 핸들
/// @param [in]	hMetric		엔진 내부 객체와의 참조 관계를 해제할 평가 핸들
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Metric_close(VHSession hSession, VHMetric hMetric);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션에서 지정된 평가 객체를 이용해 평가값들을 계산하고 그 결과를 반환합니다.
///
/// @param [in]		hSession	세션 핸들
/// @param [in]		hMetric		평가값 계산에 이용할 평가 객체 핸들
/// @param [in]		ppBuf   	평가값 계산에 이용할 신경망 추정 텐서 핸들 목록의 래핑 구조체 주소
/// @param [out]	ppLsBuf		계산 결과 구해진 결과 텐서 핸들 목록을 받아올 래핑 구조체 변수의 주소
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Metric_evaluate(VHSession hSession, VHMetric hMetric, const VExBuf* ppBuf, const VExBuf** ppLsBuf);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션에서 새로운 옵티마이저 객체를 생성하도록 하고 생성된 객체의 핸들을 반환합니다.
///
/// @param [in]		hSession	세션 핸들
/// @param [out]	phOptimizer	생성된 객체의 핸들을 받아올 변수 포인터
/// @param [out]	hParameters	생성된 옵티마이저가 최적화 대상으로 삼을 파라미터들이 담긴 파라미터군 객체의 핸들
/// @param [in]		sBuiltin	V_Session_getBuiltinNames()에서 알려주는 옵티마이저 유형 이름 중 하나로서 생성할 손실의 종류 지정
/// @param [in]		pDictBuf 	VDict 형식 옵션 정보의 래핑 구조체 주소
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Optimizer_create(VHSession hSession, VHOptimizer * phOptimizer, VHParameters hParameters, string sBuiltin, const VExBuf* pDictBuf);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션에서 지정된 옵티마이저 핸들과 객체 간의 참조 관계를 해제하고 더 이상 필요가 없는 경우 객체를 삭제합니다.
///
/// @param [in]	hSession	세션 핸들
/// @param [in]	hOptimizer 	엔진 내부 객체와의 참조 관계를 해제할 옵티마이저 핸들
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Optimizer_close(VHSession hSession, VHOptimizer hOptimizer);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션에서 지정된 옵티마이저에 각종 설정 정보를 지정합니다.
///
/// @param [in]	hSession	세션 핸들
/// @param [in]	hOptimizer 	설정 정보를 지정할 대상 옵티마이저 핸들
/// @param [in]	pDictBuf 	VDict 형식 옵티마이저 설정 정보의 래핑 구조체 주소
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Optimizer_set_option(VHSession hSession, VHOptimizer hOptimizer, const VExBuf* pDictBuf);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션에서 지정된 옵티마이저를 이용하여 역전파에서 계산되어 있는 기울기 정보를 관할 대상 파라미터들에 반영해 수정합니다.
///
/// @param [in]	hSession	세션 핸들
/// @param [in]	hOptimizer 	파라미터 갱신 대상 옵티마이저 핸들
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Optimizer_step(VHSession hSession, VHOptimizer hOptimizer);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션에서 지정된 파라미터군 핸들과 객체 간의 참조 관계를 해제하고 더 이상 필요가 없는 경우 객체를 삭제합니다.
///
/// @param [in]	hSession	세션 핸들
/// @param [in]	hParameters 엔진 내부 객체와의 참조 관계를 해제할 파라미터군 핸들
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Parameters_close(VHSession hSession, VHParameters hParameters);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션에서 지정된 파라미터군 객체가 관할하는 파라미터/기울기 텐서들을 조회하여 정보 목록과 이름 리스트 형태로 반환합니다.
///
/// @param [in]		hSession		세션 핸들
/// @param [in]		hParameters		파라미터/기울기 텐서들을 조회할 파라미터군 핸들
/// @param [in]		bGrad 		    false이면 파라미터 텐서들을, true이면 기울기 텐서들을 조회
/// @param [out]	ppListBuf		순전파 실행시 파라미터 이용 순서에 따른 목록 이름 리스트
/// @param [out]	ppDictBuf		조회된 텐서들의 목록 사전 정보의 래핑 구초체 주소를 받아올 포인터 변수
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Parameters_getWeights(VHSession hSession, VHParameters hParameters, bool bGrad, const VExBuf** ppListBuf, const VExBuf** ppDictBuf);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션에서 지정된 파라미터군 객체가 관할하는 파라미터들에 대응되는 기울기 텐서들의 내용을 모두 0으로 초기화합니다.
///
/// @param [in]		hSession		세션 핸들
/// @param [in]		hParameters		기울기 텐서 초기화를 수행할 파라미터군 핸들
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Parameters_zeroGrad(VHSession hSession, VHParameters hParameters);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션에서 지정된 파라미터군 객체가 관할하는 파라미터들에 대응되는 기울기 텐서들의 내용을 초기 상태로 변경합니다.
///
/// @param [in]		hSession		세션 핸들
/// @param [in]		hParameters		기울기 텐서 초기화를 수행할 파라미터군 핸들
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Parameters_initWeights(VHSession hSession, VHParameters hParameters);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션에서 새로운 사용자 정의 함수 객체를 생성하도록 하고 생성된 객체의 핸들을 반환합니다.
///
/// @param [in]		hSession	세션 핸들
/// @param [out]	phFunction	생성된 객체의 핸들을 받아올 변수 포인터
/// @param [in]		sBuiltin	V_Session_getBuiltinNames()에서 알려주는 사용자 정의 함수 유형 이름 중 하나로서 현재는 "user_defined"만 허용
/// @param [in]		pDictBuf 	VDict 형식 옵션 정보의 래핑 구조체 주소
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Function_create(VHSession hSession, VHFunction *phFunction, string sBuiltin, string sname, void* pCbAux, const VExBuf* pDictBuf);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션에서 지정된 사용자 정의 함수 핸들과 객체 간의 참조 관계를 해제하고 더 이상 필요가 없는 경우 객체를 삭제합니다.
///
/// @param [in]	hSession	세션 핸들
/// @param [in]	hFunction 	엔진 내부 객체와의 참조 관계를 해제할 사용자 정의 함수 핸들
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Function_close(VHSession hSession, VHFunction hFunction);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// 연결된 세션의 자원을 활용해 FFT 분석을 수행하고 얻어진 스펙트럼 정보를 반환합니다.
///
/// @param  [in]	hSession			세션 핸들
/// @param 	[in]	pwBuf				분석 대상 파형 샘플 정보를 담은 텐서 핸들을 포함하는 사전 정보의 래핑 구조체 주소
/// @param 	[in]	spec_interval   	복수의 FFT 분석 수행 대상 선정에 이용할 샘플 단위의 간격
/// @param 	[in]	freq_in_spectrum	FFT 분석 결과로 추출할 주파수의 가짓수
/// @param 	[in]	fft_width			단위 FFT 분석에 이용할 샘플 갯수
/// @param 	[out]	ppRsBuf				분석 결과 얻어진 스펙트럼 정보를 갖는 텐서 핸들을 포함하는 사전 정보의 래핑 구조체를 받아롤 포인터 주소
///
/// @returns	성공하면 VERR_OK(0), 실패하면 오류 코드 값
////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" VAPI VRetCode V_Util_fft(VHSession hSession, const VExBuf* pwBuf, int64 spec_interval, int64 freq_in_spectrum, int64 fft_width, const VExBuf** ppRsBuf);
