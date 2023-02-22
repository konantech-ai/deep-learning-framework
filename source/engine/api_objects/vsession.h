#pragma once

#include "../api/vdefine.h"

class VSessionCore;
class VModuleCore;
class VDeviceManager;
class VHyperManager;
class VCbItem;
class VCbBackInfo;
class VCbBackSlot;

class VSession {
public:
    VSession();
    VSession(const VSession& src);
    VSession(VDict kwArgs);
    VSession(VHSession hSession);
    virtual ~VSession();
    VSession& operator =(const VSession& src);

    bool operator ==(const VSession& src) const;
    bool operator !=(const VSession& src) const;

    operator VHSession();

    VHSession cloneCore();
    VHSession cloneHandle();

    void closeHandle();

    int getIdForHandle(VHandle handle);

    string getVersion();

    void seedRandom(int64 rand_seed);

    void registCallbackSlot(VCbBackSlot slot);
    void registUserDefinedFunction(string sName, VFunction function);

    void setFunctionCbHandler(void* pCbAux, VCbForwardFunction* pFuncForward, VCbBackwardFunction* pFuncBackward, VCbClose* pCbClose);
    VFuncCbHandlerInfo& getFunctionCbHandlerInfo();

    bool lookupUserDefinedFunctions(string opCode, VValue* pFunction);

    void closeObjectInfo();

    bool getNoGrad();
    bool getNoTracer();

    bool isTracerDirty() { return false; }

    void setNoGrad(bool noGrad);
    void setNoTracer(bool no_tracer);

    VDeviceManager device_man();
    VHyperManager hyper_man();

    void registMacro(string macroName, VModule module, VDict kwArgs);
    VModule getMacro(string macroName);

    VTensor util_fft(VTensor wave, int64 spec_interval, int64 freq_in_spectrum, int64 fft_width);

    int addForwardCallbackHandler(VCbForwardModule* pCbFunc, VCbClose* pCbClose, VDict filters, VDict instInfo);
    int addBackwardCallbackHandler(VCbBackwardModule* pCbFunc, VCbClose* pCbClose, VDict filters, VDict instInfo);
    void removeCallbackHandler(int nId);

    bool needCallback();

    void invokeMatchingCallbacks(
        VModuleCore* pStarter, string name, VTensorDict xs, VTensorDict ys, VTensorDict sideTerms, VParameters params,
        bool train, bool noGrad, int nDevice, bool bPre, VCbBackInfo cbInfo, VExecTracer tracer);

    void invokeCallback(
        VModuleCore* pStarter, VCbItem item, string name, bool train, int nDevice, bool bPre,
        bool noGrad, VTensorDict xs, VTensorDict ys, VTensorDict sideTerms, VParameters params, VExecTracer tracer);

    VDict getLeakInfo(bool sessionOnly);

protected:
    float* m_fft_core_split(float* pffts, float* pbuf1, float* pbuf2, int64 fft_width, int64 bsize);

protected:
    static string ms_sEngineVersion;

    VSessionCore* m_core;

public:
    void RegistCustomModuleExecFunc(VCbCustomModuleExec* pFunc, void* pInst, void* pAux);
    void RegistFreeReportBufferFunc(VCbFreeReportBuffer* pFunc, void* pInst, void* pAux);

    VCbCustomModuleExec* getCustomModuleExecCbFunc(void** pInst, void** pAux);
    VCbFreeReportBuffer* getFreeReportBufferCbFunc(void** pInst, void** pAux);

    void SetLastError(VException ex);
    VRetCode GetLastErrorCode();
    VList GetLastErrorMessageList();
};
