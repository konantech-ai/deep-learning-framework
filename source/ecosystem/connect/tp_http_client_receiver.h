#pragma once

#include "../utils/tp_common.h"

class HttpClientReceiver {
public:
	HttpClientReceiver();
	virtual ~HttpClientReceiver();

    void registCustomModuleExecFunc(VCbCustomModuleExec* pFunc, void* pInst, void* pAux);
	void registFreeReportBufferFunc(VCbFreeReportBuffer* pFunc, void* pInst, void* pAux);

protected:
};
