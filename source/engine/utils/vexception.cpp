#include "vexception.h"

VException::VException() {
	m_core = new VExceptionCore();
}

VException::VException(VExceptionCore* core) {
	m_core = core->clone();
}

VException::VException(const VException& src) {
	m_core = src.m_core->clone();
}

VException::VException(int nErrCode, string file, int line) {
	if (nErrCode == 0) return;

	m_core = new VExceptionCore();

	std::replace(file.begin(), file.end(), '\\', '/');

	m_core->m_nErrCode = nErrCode;
	m_core->m_msgStack.push_back(m_createMessage(nErrCode, file, line));
}

VException::VException(int nErrCode, string sParam, string file, int line) {
	m_core = new VExceptionCore();

	std::replace(file.begin(), file.end(), '\\', '/');

	m_core->m_nErrCode = nErrCode;
	m_core->m_msgStack.push_back(m_createMessage(nErrCode, file, line, sParam));
}

VException::VException(int nErrCode, string sParam1, string sParam2, string file, int line) {
	m_core = new VExceptionCore();

	std::replace(file.begin(), file.end(), '\\', '/');

	m_core->m_nErrCode = nErrCode;
	m_core->m_msgStack.push_back(m_createMessage(nErrCode, file, line, sParam1, sParam2));
}

VException::VException(int nErrCode, string sParam1, string sParam2, string sParam3, string file, int line) {
	m_core = new VExceptionCore();

	std::replace(file.begin(), file.end(), '\\', '/');

	m_core->m_nErrCode = nErrCode;
	m_core->m_msgStack.push_back(m_createMessage(nErrCode, file, line, sParam1, sParam2, sParam3));
}

VException::VException(int nErrCode, const VException& src, string file, int line) {
	std::replace(file.begin(), file.end(), '\\', '/');

	m_core = src.m_core->clone();
	m_core->m_msgStack.push_back(m_createMessage(nErrCode, file, line));
}

VException::VException(int nErrCode, const VException& src, string sParam1, string file, int line) {
	std::replace(file.begin(), file.end(), '\\', '/');

	m_core = src.m_core->clone();
	m_core->m_msgStack.push_back(m_createMessage(nErrCode, file, line, sParam1));
}

VException::~VException() {
	m_core->destroy();
}

VException& VException::operator =(const VException& src) {
	if (&src != this) m_core = src.m_core->clone();
	return *this;
}

VRetCode VException::GetErrorCode() {
	return m_core->m_nErrCode;
}

string VException::m_createMessage(int nErrCode, string file, int line, string sParam1, string sParam2, string sParam3) {
	char buffer[1024];

	const char* p1 = (sParam1.length() > 0) ? sParam1.c_str() : "";
	const char* p2 = (sParam2.length() > 0) ? sParam2.c_str() : "";
	const char* p3 = (sParam3.length() > 0) ? sParam3.c_str() : "";

	ms_createMessage(buffer, nErrCode, p1, p2, p3);

	snprintf(buffer + strlen(buffer), 1024 - strlen(buffer), " in %s:%d", file.c_str(), line);

	return (string)buffer;
}

VList VException::GetErrorMessageList() {
	if (m_core->m_nErrCode == 0) return VList();
	return m_core->m_msgStack;
}

VList VException::GetErrorMessageList(VRetCode nErrorCode) {
	return VList({ "Sorry, error message for static code is not implemented yet..." });
}

void VException::ms_createMessage(char *pBuf, int nErrCode, const char* p1, const char* p2, const char* p3) {
	switch (nErrCode) {
	case VERR_INTERNAL_LOGIC:
		snprintf(pBuf, 1024, "엔진내부의 논리적 오류: %s", p1);
		break;
	case VERR_SHAPE_CONV2D:
		snprintf(pBuf, 1024, "입력 형상의 채널수가 Conv2D 레이어의 xchn 값과 부합되지 않습니다.");
		break;
	case VERR_PARALLEL_EVALUATION:
		snprintf(pBuf, 1024, "병렬 처리 과정에서 예외가 발생하였습니다.");
		break;
	case VERR_MACRO_UNREGISTERED:
		snprintf(pBuf, 1024, "등록되지 않은 매크로 이름 %s이(가) 사용되었습니다.", p1);
		break;
	case VERR_SHAPE_NOT_2D_DIMENSION:
		snprintf(pBuf, 1024, "형상으로 지정된 %s 값이 2차원 형상이 아닙니다.", p1);
		break;
	case VERR_BAD_SHAPE_TENSOR:
		snprintf(pBuf, 1024, "%s() 함수에 사용된 텐서의 형상이 처리 과정에 부합되지 않습니다.", p1);
		break;
	case VERR_UNMATCHED_SHAPE_IN_CROSSENTROPY:
		snprintf(pBuf, 1024, "크로스엔트로피 연산을 위한 두 성분의 형상이 처리 과정에 부합되지 않습니다.");
		break;
	case VERR_BAD_PADDING_DIMENSION:
		snprintf(pBuf, 1024, "패딩 형상으로 지정된 값이 4차원 형상이 아닙니다.");
		break;
	case VERR_BAD_PADDING_ARGUMENT:
		snprintf(pBuf, 1024, "패딩 값으로 알 수 없는 내용인 %s 값이 지정되었습니다.", p1);
		break;
	case VERR_UNKNOWN_PADDING_MODE:
		snprintf(pBuf, 1024, "패딩 모드 값으로 알 수 없는 내용인 %s 값이 지정되었습니다.", p1);
		break;
	case VERR_LAYER_EXEC_INSHAPE_MISMATCH:
		snprintf(pBuf, 1024, "%s 레이어 입력 형상이 %s로 설정되었지만 %s 형상의 데이터가 주어졌습니다.", p1, p2, p3);
		break;
	case VERR_LAYER_EXEC_OUTSHAPE_MISMATCH:
		snprintf(pBuf, 1024, "%s 레이어 출력 형상이 %s로 설정되었지만 %s 형상의 데이터가 생성되었습니다.", p1, p2, p3);
		break;
	case VERR_WILL_BE_IMPLEMENTED:
		snprintf(pBuf, 1024, "아직 구현 예정인 %s 기능이 사용되었습니다.", p1);
		break;
	case VERR_INDEXING_ON_NULL_SHAPE:
		snprintf(pBuf, 1024, "내용이 지정되지 않은 형상 변수에 대한 원소 접근이 시도되었습니다.");
		break;
	case VERR_HOSTMEM_ALLOC_FAILURE:
		snprintf(pBuf, 1024, "호스트 메모리 블록의 할당에 실패했습니다. 메모리 사용량을 확인하세요.");
		break;
	case VERR_PARAMETER_IS_NOTALLOCATED:
		snprintf(pBuf, 1024, "파라미터에 대한 메모리 할당이 이루어지지 않은 상태여서 모듈 실행이 불가능합니다.");
		break;
	case VERR_BAD_GROUP_IN_CONV2D_LAYER:
		snprintf(pBuf, 1024, "컨볼루션 레이어에 잘못된 group 값(%s mod %s is not zero)이 지정되었습니다.", p1, p2);
		break;
	case VERR_BAD_HEAD_CNT_FOR_MH_ATTENTION:
		snprintf(pBuf, 1024, "멀티헤드 어텐션의 벡터 크기에 부합되지 않는 헤드수입니다.");
		break;
	case VERR_BAD_KQV_SHAPE_FOR_MH_ATTENTION:
		snprintf(pBuf, 1024, "멀티헤드 어텐션의 키, 쿼리, 밸류 성분의 형상이 일치하지 않습니다.");
		break;
	case VERR_PRUNING_WITHOUT_NAME:
		snprintf(pBuf, 1024, "이름 지정 없는 pruning 분기가 사용되었습니다. 의도적인 처리라면 drop 값을 true로 주세요.");
		break;
	case VERR_GETNAME_NOT_FOUND:
		snprintf(pBuf, 1024, "%s 레이어에서 get 속성으로 지정된 %s 값에 해당하는 텐서를 찾을 수 없습니다.", p1, p2);
		break;
	case VERR_LOSS_TERM_NOT_FOUND:
		snprintf(pBuf, 1024, "손실 함수 정의에 지정된 %s 이름의 텐서 성분을 찾을 수 없습니다.", p1);
		break;
	case VERR_BAD_SHAPE_FOR_IMAGE_LAYER:
		snprintf(pBuf, 1024, "이미지 처리용 %s 계층에 부합되지 않는 형상 %s이 주어졌습니다.", p1, p2);
		break;
	case VERR_RECURSIVE_MODULE_STRUCTURE:
		snprintf(pBuf, 1024, "모듈 %s와 모듈 %s 사이에 순환 구조를 만드는 부모-자식 관계 설정이 요구되었습니다.", p1, p2);
		break;
	case VERR_UNMATCH_ON_TENSOR_DATATYPE:
		snprintf(pBuf, 1024, "%s 타입의 텐서가 필요한 처리에 %s 타입 텐서가 잘못 제공되었습니다.", p1, p2);
		break;
	case VERR_MODULE_EXPAND_FAILURE:
		snprintf(pBuf, 1024, "%s 타입의 모듈을 전개하는 과정에서 발생하였습니다.", p1);
		break;
	case VERR_UNKNWON_SHAPE_FOR_GET_FIELD:
		snprintf(pBuf, 1024, "get 필드로 지정된 %s 입력 텐서의 형상이 제공되지 않았습니다.", p1);
		break;
	default:
		snprintf(pBuf, 1024, "오류메시지가 준비되지 않았습니다.(code:%d,%s,%s)", nErrCode, p1, p2);
		break;
	}
}