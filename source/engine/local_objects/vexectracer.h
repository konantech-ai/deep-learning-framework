#pragma once

#include "../api/vcommon.h"
#include "../api_objects/vsession.h"

enum class VMathFunc;
enum class TensorCloneInit;
class VCbBackSlotCore;
class  VFunctionCore;

class VFuncArg;

typedef vector<VFuncArg> VFuncArgList;

class VExecTracerCore;

class VExecTracer {
public:
	VExecTracer();
	VExecTracer(VSession session, string sBuiltin, VDict kwArgs = {});
	VExecTracer(const VExecTracer& src);
	VExecTracer(VExecTracerCore* core);
	virtual ~VExecTracer();
	VExecTracer& operator =(const VExecTracer& src);
	VExecTracerCore* getClone();
	VExecTracerCore* getCore();
	void destroyCore();
	VSession session() const;
	bool isValid();
	int getRefCnt();
	int getNth();
protected:
	VExecTracerCore* m_core;
public:
	bool hasValidHistory(VTensorDict xs);	// 유효 표시가 살아있고 저장된 입력과 동일한 텐서 구성일 때 한하여 true 반환
	
	void closeRecording(VTensorDict ys);
	void reset();
	void removeBranch();

	//VExecTracerCore* getCore();
	//VExecTracerCore* cloneCore();

	VTensor createTensor(VSession session, VShape shape, VDataType type, int nDevice);
	VTensor createTensor(VSession session, VHTensor hTensor);
	VTensor createTensor(VSession session, VTensorCore* core);
	VTensor createTensor(VTensor src, VShape shape, TensorCloneInit init);

	void whatToDo(); // 고민이 필요한 부분
	void setInput(VTensorDict xs);	// 전체를 리셋하고 새로 기록 시작
	//void bookeepThreadCall(int nThread);
	void addMathCall(VMathFunc func, VFuncArgList args);
	void addInvokeForwardCallback(VCbForwardModule* pCbFunc, VCbClose* pCbClose, VDict instInfo, VDict statusInfo, VDict tensorDict);
	void addInvokeBackwardCallback(VCbBackwardModule* pCbFunc, VCbClose* pCbClose, VDict instInfo, VDict statusInfo, VDict tensorDict, VDict gradDict);

	void addCallForwardUDF(VFunctionCore* functor, int nInst, VTensor y, VTensorList operands, VDict opArgs);
	void addCallBackwardUDF(int nInst, VTensor y, VTensor ygrad, VTensor xgrad, int nth);

	VTensorDict executeHistory();

	void openBranch(int nDivisions);
	void setVoidBranch(int nth);
	
	VExecTracer setValidBranch(int nth, string name, VTensorDict xs);

	void setFork(int nDivisions);

	void addTensor(VTensor tensor);
	void dump(VList tids);

	void dumpHistoryForDebug();

};
