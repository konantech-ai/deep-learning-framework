#pragma once

#include "../api/vcommon.h"
#include "../api_objects/vtensor.h"
#include "../api_objects/vsession.h"
#include "../local_objects/vexectracer.h"

class VBackQueue {
public:
	VBackQueue(VSession session, VExecTracer tracer);
	~VBackQueue();

	// 1. node가 pm 타입이면 바로 backprop() 호출해 처리하고 종료
	// 2. node 번호가 m_GradMap에 있는지 검사해 있으면 누산, 없으면 등록
	// 3. node의 --opndRef# 수행하고 이 값이 0 보다 크면 처리 종료
	//    3-1. 0 이하이고 병렬 처리 강 건너 온 경우 아니면 m_freeQueue 등록
	//    3-2. 병렬 처리 강 건너 온 경우면 m_crossStack 등록
	// => 강 건너 온 친구를 저장할 별도 메소드를 추가해 분리함

	void regist(VTensor node, VTensor grad);
	void pushCross(VTensor node, VTensor grad);

	// 손실 계산에서 배제된 끈 떨어진 신경망 출력 텐서 등록.
	// 이들을 처리에서 누락시킬 경우 참조 계수 감소에 반영되지 않으면서
	// 처리 대상 오퍼랜드들이 regist()에서 등록이 안 되어 작업이 잘못 중단되는 사태가 발생함
	void registCrossNodes(VTensorDict nodes);

	// 역전파 처리가 불필요한 텐서 등록
	// 이들을 처리에서 누락시킬 경우 참조 계수 감소에 반영되지 않으면서
	// 처리 대상 오퍼랜드들이 regist()에서 등록이 안 되어 작업이 잘못 중단되는 사태가 발생함
	void registNoGrad(VTensor node);

	// m_freeQueue.size() == 0, m_crossStack.size() == 0이면 true
	// 단 이때 m_GradMap.size() != 0이면 예외 발생
	bool isEnd();

	// 1. m_freeQueue.size() == 0이고 m_crossStack.size() == 0이면 예외 발생
	//    1-1. m_crossStack.size() > 0이면 병렬 디바이스별로 일괄 강 건너기
	// 2. m_freeQueue 선두를 pop해 작업 대상으로 삼음
	// 3. m_GradMap에서 대응되는  항목 찾아 기울기정보 꺼내고 항목 삭제
	// 4. 작업대상 텐서의 역전파 함수 호출
	// 5. 작업대상 오퍼랜드들 중 no_grad 아닌 것은 작업 결과  얻어진 기울기와 함께 push() 호출 필수
	void step();

protected:
	void m_dump(string action);
	void m_execute_parallel_queues(VExecTracer tracer);

	static void ms_branchMain(void* aux);

protected:
	VSession m_session;

	int m_nInstanceId;

	VTensorList m_freeStack;
	VTensorList m_emptyStack;
	VTensorMap m_gradMap;

	map<int, int> m_RefCnt; // debugging용 임시 변수

	VTensorList m_crossStack;	// 강 건너 온 동무들: parallel 처리 종료 때 건너옴, 역전파 병렬 처리 위해 몰아서 보내야 함
	VTensorMap m_crossedMap;

	string m_lastOpName;

	VExecTracer m_execTracer;

public:
	static bool ms_bQueueTrace;
	static int ms_nNextInstanceId;
};
