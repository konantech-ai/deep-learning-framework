#pragma once

#include "../api/vcommon.h"
#include "../api_objects/vsession.h"

class VFunctionCore;

class VFunction {
public:
    VFunction();
    VFunction(VSession session, string sBuiltin, VDict kwArgs = {});
    VFunction(const VFunction& src);
    VFunction(VSession session, VHFunction handle);
    VFunction(VFunctionCore* core);
    virtual ~VFunction();
    VFunction& operator =(const VFunction& src);
    VHFunction cloneCore();
    VHFunction cloneHandle();
    VFunctionCore* getClone();
    VFunctionCore* getCore();
    bool isValid();
    void closeHandle();
    VSession session();
    int getNth();
    int getRefCnt();
    void incRefCount();
protected:
    VFunctionCore* m_core;
public:

protected:
    string m_sName;
    void* m_pCbAux;
public:
    VFunction(VSession session, string sBuiltin, string sName, void* pCbAux, VDict kwArgs);

    VTensor forward(int nInst, VTensorList operands, VDict opArgs);
    VTensor backward(int nInst, VTensor ygrad, int nth, VTensorList operands, VDict opArgs);

    static VList GetBuiltinNames();

protected:
    static VStrList ms_builtin;
};
