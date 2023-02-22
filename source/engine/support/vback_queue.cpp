#include "../support/vback_queue.h"
#include "../support/vmath.h"
#include "../api_objects/vtensor.h"
#include "../api_objects/vsession.h"
#include "../local_objects/vexectracer.h"
#include "../local_objects/vexectracer_core.h"
#include "../local_objects/vdevicemanager.h"

bool VBackQueue::ms_bQueueTrace = false;
int VBackQueue::ms_nNextInstanceId = 0;

VBackQueue::VBackQueue(VSession session, VExecTracer tracer) {
	m_session = session;
	m_lastOpName = "(init)";
	m_execTracer = tracer;
	m_nInstanceId = ms_nNextInstanceId++;
}

VBackQueue::~VBackQueue() {
}

void VBackQueue::regist(VTensor node, VTensor grad) {
	m_execTracer.addTensor(node);
	m_execTracer.addTensor(grad);

	node.invokeBackpropCallback(grad, m_execTracer);

	if (node.is_pm()) {
		node.pm_backprop(grad, m_execTracer);
		if (ms_bQueueTrace) {
			if (0) printf("[QUEUE pm] T#%d processed\n", node.getNth());
		}
		return;
	}
	else {
		int tid = node.getNth();
		int refCnt = node.decOperandRefCount();

		if (m_gradMap.find(tid) == m_gradMap.end()) {

			if (0) printf("m_gradMap[T#%d, refCnt:%d] = grad(T#%d) registered\n", tid, refCnt, grad.getNth());

			m_gradMap[tid] = grad;
			m_RefCnt[tid] = refCnt;
		}
		else {
			m_gradMap[tid].accGrad(grad, m_execTracer);
			m_RefCnt[tid] = refCnt;

			if (0) printf("m_gradMap[T#%d, refCnt:%d] = grad(T#%d) added onto T#%id\n", tid, refCnt, grad.getNth(), m_gradMap[tid].getNth());
		}

		if (refCnt <= 0) {
			m_freeStack.push_back(node);
		}
	}

	m_dump("regist");
}

void VBackQueue::registNoGrad(VTensor node) {
	m_execTracer.addTensor(node);

	if (node.is_pm()) {
		if (0) printf("[PM] import pm#%d without grad\n", node.getNth());
		return;
	}
	else {
		int tid = node.getNth();
		int refCnt = node.decOperandRefCount();

		if (0) printf("\t[T#%d(ref:%d): %s] added without grad\n", tid, refCnt, node.shape().desc().c_str());

		if (refCnt <= 0) {
			if (m_gradMap.find(tid) == m_gradMap.end()) {
				if (tid == 140274) {
					if (0) printf("node[T#%d] added onto empty stack\n", tid);
				}
				m_emptyStack.push_back(node);
			}
			else {
				if (tid == 140274) {
					if (0) printf("node[T#%d] added onto free stack\n", tid);
				}
				m_freeStack.push_back(node);
			}
		}
	}

	m_dump("registNoGrad");
}

void VBackQueue::registCrossNodes(VTensorDict nodes) {
	m_crossedMap.clear();

	for (auto& it : nodes) {
		VTensor opnd = it.second.getNthOperand(0);
		m_crossedMap[opnd.getNth()] = opnd;
		m_execTracer.addTensor(opnd);
	}
	m_dump("registCrossNodes");
}

void VBackQueue::pushCross(VTensor node, VTensor grad) {
	int tid = node.getNth();
	int refCnt = node.decOperandRefCount();

	if (m_gradMap.find(tid) == m_gradMap.end()) {
		m_gradMap[tid] = grad;
		m_RefCnt[tid] = refCnt;

		if (0) printf("m_gradMap[T#%d, refCnt:%d] = grad(T#%d) registered by pushCross\n", tid, refCnt, grad.getNth());
	}
	else {
		if (0) printf("m_gradMap[T#%d, refCnt:%d] = grad(T#%d) ignored in pushCross\n", tid, refCnt, grad.getNth());
		// GAN에서처럼 eval() 모드로 계산된 성분의 경우
		return;
		//VP_THROW1(VERR_INTERNAL_LOGIC, __func__);
	}

	if (refCnt <= 0) {
		if (0) printf("m_gradMap[T#%d, refCnt:%d] with grad(T#%d) pushed into cross stack\n", tid, refCnt, grad.getNth());
		m_crossStack.push_back(node);
		// tensor.backward() 형식의 호출에서는 m_crossedMap 정보가 설정되지 않으며
		// 단일 출력에 대한 처리이므로 무시할 수 있을 듯...
		if (m_crossedMap.find(node.getNth()) != m_crossedMap.end()) {
			if (0) printf("m_gradMap[T#%d, refCnt:%d] with grad(T#%d) deleted from cross map\n", tid, refCnt, grad.getNth());
			m_crossedMap.erase(m_crossedMap.find(node.getNth()));
		}
	}
	else {
		// 디버깅중: refCnt가 남아 있으며 다른 처리에 의해 감소하는지 보기로 한다.
		//VP_THROW(VERR_INTERNAL_LOGIC_ERROR);
	}

	m_dump("pushCross");
}

bool VBackQueue::isEnd() {
	if (m_freeStack.size() > 0) return false;
	if (m_emptyStack.size() > 0) return false;
	if (m_crossStack.size() > 0) return false;

	if (m_gradMap.size() > 0) {
		// custom loss 기술 과정에서 loss 계산에 사용되지 않는 수식이 있는 경우 역전파 경로에서 배제되어 큐에서 해소되지 못하는 현상이 생긴다.
		// Yolo4 예제에서처럼 loss, metric 계산에 이용되는 공통 수식을 이용하다보면 이런 경우가 생길 수 있다.
		m_gradMap.clear();
	}

	return true;
}

void VBackQueue::step() {
	if (m_freeStack.size() == 0) {
		if (m_emptyStack.size() > 0) {
			VTensor tensor = m_emptyStack.back();
			m_emptyStack.pop_back();
			tensor.backprop_noGgrad(this);
			m_dump("backprop_noGgrad");
		}
		else {
			if (m_crossStack.size() == 0) VP_THROW1(VERR_INTERNAL_LOGIC, __func__);
			m_execute_parallel_queues(m_execTracer);
			m_dump("m_execute_parallel_queues");
		}
		return;
	}

	VTensor tensor = m_freeStack.back();
	VTensor grad = m_gradMap[tensor.getNth()];

	string tshape = tensor.shape().desc();
	string gshape = grad.isValid() ? grad.shape().desc() : "empty";

	m_lastOpName = tensor.getOpName();
	m_freeStack.pop_back();
	m_gradMap.erase(m_gradMap.find(tensor.getNth()));
	m_RefCnt.erase(m_RefCnt.find(tensor.getNth()));

	tensor.backprop(grad, this, m_execTracer);

	m_dump("backprop");
}

void VBackQueue::m_execute_parallel_queues(VExecTracer tracer) {
	int nDevCnt = m_session.device_man().getDeviceCount();

	if (nDevCnt == 0) VP_THROW1(VERR_INTERNAL_LOGIC, __func__);

	for (auto& it : m_crossedMap) {
		it.second.decOperandRefCount();
	}

	tracer.openBranch(nDevCnt);

	VList contexts;

	int64 nFrom = 0;
	int64 nCount = 0;

	for (int nDevice = 0; nDevice < nDevCnt; nDevice++) {
		VTensorMap tensors;
		VTensorMap grads;
		
		for (auto& it : m_crossStack) {
			VTensor tensor = it;

			if (tensor.device() == nDevice) {
				int tid = tensor.getNth();
				
				VTensor grad = m_gradMap[tid];

				if (0) printf("m_gradMap[T#%d] with grad(T#%d) used in m_execute_parallel_queues()\n", tid, grad.getNth());

				if (grad.shape() != tensor.shape() || grad.device() != tensor.device()) {
					if (nCount == 0) nCount = tensor.shape()[0];
					else if (nCount != tensor.shape()[0]) VP_THROW1(VERR_INTERNAL_LOGIC, __func__);
				
					grad = grad.getSlicePiece(nFrom, nCount, nDevice, tracer);
				}

				tensors[tid] = tensor;
				grads[tid] = grad;
			}
		}

		for (auto& it : m_crossedMap) {
			VTensor tensor = it.second;

			if (tensor.device() == nDevice) {
				int tid = tensor.getNth();
				tensor.incOperandRefCount();

				VTensor grad = tracer.createTensor(tensor, tensor.shape(), TensorCloneInit::zeros);
				grad.setZero(tracer);	// tracer 반복 수행 때마다 초기화되어야 하므로 생성과 관계 없이 추가 호출 필요

				tensors[tid] = tensor;
				grads[tid] = grad;
			}
		}

		nFrom += nCount;
		nCount = 0;

		if (tensors.size() == 0) {
			contexts.push_back(VDict());
			tracer.setVoidBranch(nDevice);
		}
		else {
			VExecTracer childTracer = tracer.setValidBranch(nDevice, "backward_branch", vutils.toTensorDict(m_session, tensors));

			VDict ctx{
				{"tensors", vutils.toMapInternal(tensors)}, {"grads", vutils.toMapInternal(grads)},
				{"session", m_session.cloneCore()},  {"device", nDevice}, {"tracer", (VObjCore*)childTracer.getClone()} };

			contexts.push_back(ctx);
		}
	}

	//if (nFrom != 0 && nFrom != m_gradMap.begin()->second.shape()[0]) VP_THROW1(VERR_INTERNAL_LOGIC, __func__);

	tracer.setFork(nDevCnt);

	std::thread** ppThreads = new std::thread * [nDevCnt];

	for (int nDevice = 0; nDevice < nDevCnt; nDevice++) {
		ppThreads[nDevice] = NULL;

		VDict ctx = contexts[nDevice];
		if (ctx.size() == 0) continue;

		ppThreads[nDevice] = new std::thread(ms_branchMain, ctx.cloneCore());
	}

	m_crossStack.clear();
	m_gradMap.clear();
	m_RefCnt.clear();

	string failReport;

	for (int nDevice = 0; nDevice < nDevCnt; nDevice++) {
		if (ppThreads[nDevice] == NULL) continue;

		ppThreads[nDevice]->join();

		VDict ctx = contexts[nDevice];

		int errcode = vutils.seek_dict(ctx, "errcode", 0);

		if (errcode != 0) {
			VException exInfo((VExceptionCore*)(VObjCore*)ctx["errinfo"]);
			VP_THROW1(VERR_PARALLEL_EVALUATION, exInfo);
			//failReport += "thread-" + to_string(nDevice) + ": " + errcode + "\nDevice";
		}

		ctx.freeClone();
	}

	delete[] ppThreads;

	if (failReport != "") VP_THROW1(VERR_PARALLEL_BACKWARD, failReport);

}

void VBackQueue::ms_branchMain(void* aux) {
	VDict ctx((VDictCore*)aux);

	try {
		VSession session((VHSession)ctx["session"]);
		VExecTracer tracer((VExecTracerCore*)(VObjCore*)ctx["tracer"]);

		int nDevice = ctx["device"];

		session.device_man().setCurDevice(nDevice, tracer);

		VBackQueue branchQueue(session, tracer);

		VTensorMap tensors = vutils.toTensorMap(session, ctx["tensors"]);
		VTensorMap grads = vutils.toTensorMap(session, ctx["grads"]);

		for (auto& it : tensors) {
			branchQueue.regist(it.second, grads[it.first]);
		}

		branchQueue.m_dump("backprop");

		while (!branchQueue.isEnd()) {
			branchQueue.step();
		}

		tracer.closeRecording({});
	}
	//catch (ValException vex) { ctx["errcode"] = VERR_UNDEFINED;  ex= ctx["errinfo"] = ; }
	catch (VException ex) {
		ctx["errcode"] = ex.GetErrorCode();
		ctx["errinfo"] = (VObjCore*)ex.cloneCore();
	}
	catch (...) {
		ctx["errcode"] = VERR_PARALLEL_EVALUATION;
		VException ex(VERR_PARALLEL_EVALUATION, __FILE__, __LINE__);
		ctx["errinfo"] = (VObjCore*)ex.cloneCore();
	}
}

void VBackQueue::m_dump(string action) {
	if (!ms_bQueueTrace) return;

	printf("[QUEUE %d, action:%s, last_op:%s]", m_nInstanceId, action.c_str(), m_lastOpName.c_str());

	printf("\n    [FREE STACK] ");
	for (auto& it : m_freeStack) {
		printf(" T#%d%s", it.getNth(), it.shape().desc().c_str());
	}

	printf("\n    [GRAD MAP]   ");
	for (auto& it : m_gradMap) {
		if (it.second.isValid()) {
			printf(" T#%d%s(T#%d,%d)", it.first, it.second.shape().desc().c_str(), it.second.getNth(), m_RefCnt[it.first]);
		}
		else {
			printf(" T#%d(-)", it.first);
		}
	}

	printf("\n    [EMPTY STACK]");
	for (auto& it : m_emptyStack) {
		printf(" T#%d%s", it.getNth(), it.shape().desc().c_str());
	}

	printf("\n    [CROSS STACK]");
	for (auto& it : m_crossStack) {
		printf(" T#%d%s", it.getNth(), it.shape().desc().c_str());
	}

	printf("\n    [CROSSED MAP]");
	for (auto& it : m_crossedMap) {
		printf(" [T#%d]T#%d%s", it.first, it.second.getNth(), it.second.shape().desc().c_str());
	}

	printf("\n\n");
}
