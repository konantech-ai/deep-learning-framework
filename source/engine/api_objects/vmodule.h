#pragma once

#include "../api/vcommon.h"
#include "../api_objects/vsession.h"
#include "../api_objects/vtensor.h"
#include "../api_objects/vparameters.h"
#include "../local_objects/vgraph.h"

class VBackQueue;
class VModuleCore;

class VModule {
public:
    VModule();
    VModule(VSession session, string sBuiltin, VDict kwArgs = {});
    VModule(const VModule& src);
    VModule(VSession session, VHModule handle);
    VModule(VModuleCore* core);
    virtual ~VModule();
    VModule& operator =(const VModule& src);
    VHModule cloneCore();
    VHModule cloneHandle();
    VModuleCore* getClone();
    VModuleCore* getCore();
    bool isValid();
    void closeHandle();
    VSession session();
    int getNth();
    int getRefCnt();
    void incRefCount();
protected:
    VModuleCore* m_core;
public:
    VModule(VModule& src, bool copyChildren);
    VModule(VSession session, string sBuiltin, string sMacroName, VDict kwArgs); // for "macro"
    VModule(VSession session, string sName, string sFormula, VDict paramInfo, VDict kwArgs);
    VModule(VSession session, VDict moduleInfo);

    void setName(string sName);

    string getName();
    string getBuiltIn();
    
    VModuleType getModuleType();
    int64 getParamSize();
    VDict getKwArgs();
    VShape getInShape();
    VShape getOutShape();
    VShape getExpandShape();

    void appendChild(VModule child);
    bool isDesendent(VModule module);

    void macroExpandCheck();
    
    VModule expand(VShape shape, VDict kwArgs);
    //VModule expandMacro(VShape shape, VDict kwArgs);
    VModule toDevice(string device);
    
    void setDevice(string device);

    bool isUsingCpu();

    VList params();

    VParameters getParameters();

    void loadParameters(string filePath, string mode);
    void setParamater(VDict tHandles, string mode);

    void copyChildren(VModule srcModule);
    VModule fetchChild(string name, bool bChildOnly);
    VModuleList getChildrenModules();

    VTensor evaluate(bool train, VTensor x);
    VTensorDict evaluateEx(bool train, VTensorDict xs);

    //void backward_parallel(VTensor ygrad, VTensorList operands, VBackQueue* pQueue);
    void parallel_backprop(VTensor ygrad, VTensorList operands, VBackQueue* pQueue, VExecTracer tracer);

    VGraph getGraphTemplate();

    void setMacroArgs(VDict kwArgs);

    VDict getSerializeInfo(string format);
    //void setSerializeInfo(string format, VDict info);

    int addForwardCallbackHandler(VCbForwardModule* pCbFunc, VCbClose* pCbClose, VDict filters, VDict instInfo);
    int addBackwardCallbackHandler(VCbBackwardModule* pCbFunc, VCbClose* pCbClose, VDict filters, VDict instInfo);

    void removeCallbackHandler(int nId);

    void uploadDataIndex(VList dataIdx);
    
    void m_setDevice(string device);

    void m_setAllocate(bool allocate);

    static string GetLayerFormula(string sBuiltin);

    static VList GetBuiltinCustomNames();
    //static VList GetBuiltinModelNames();
    static VList GetBuiltinLayerNames();
    static VList GetBuiltinNetworkNames();

protected:
    friend class VModuleCore;

    VModule m_cloneModuleBody();
    VModule m_expandMacroBody(VDict kwArgs);

    VDict m_mergeMacroArgs(VDict formalArgs, VDict& actualArgs);
    VTensorDict m_evaluate(VModuleCore* pStarter, VTensorDict xs, bool train, bool noGrad, int nDevice, VTensorDict& sideTerms, VExecTracer tracer);

    void m_loadParameters(FILE* fid, VList params);
    void m_loadParameters(FILE* fid, VDict params);

    static void ms_evaluateMain(void* aux);

    //static void ms_backwardMain(void* aux);

};
