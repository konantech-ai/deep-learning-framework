#pragma once

#include "../api/vcommon.h"
#include "../api_objects/vsession.h"
#include "../support/vback_queue.h"
#include "../api_objects/vparameters.h"
#include "../local_objects/vgraph.h"
#include "../local_objects/vexectracer.h"
#include "../local_objects/vexectracerpool.h"
#include "../local_objects/vcbitem.h"
#include "../local_objects/vcbbackinfo.h"

class VModuleCore : public VObjCore {
protected:
    friend class VModule;
protected:
    VModuleCore(VSession session, string sBuiltin = "", VDict kwArgs = {});
    VModuleCore* clone() { return (VModuleCore*)clone_core(); }
    ~VModuleCore();
    void m_onCreate();
    void m_onDelete();
    VSession session() { return m_session; }
protected:
    VSession m_session;
    string m_sBuiltin;
    VDict m_propDict;
    int m_nCheckCode;
    static int ms_nCheckCode;

protected:
    VModuleType m_moduleType;

    string m_sName;
    string m_sDevice;
    string m_sMacroName;

    bool m_bMacroExpanded;
    bool m_bIncludingMacro;

    VModuleList m_children;

    VDict m_macroArgs;

    VList m_params;
    int64 m_nParamSize;
    
    VShape m_shapeExpanded;

    VShape m_inShape;
    VShape m_outShape;

    string m_setName;
    string m_getName;

    VGraph m_graphTemplate;
    VParameters m_parameters;

    map<int, VGraph> m_graphMap;

    bool m_bNeedForwardCallbeck;
    bool m_bNeedBackwardCallbeck;

    map<int, VCbItem> m_cbForwardItemMap;
    map<int, VCbItem> m_cbBackwardItemMap;

    VList m_dataIdx;
    VList m_dataIdxes;

    string m_formula;
    VDict m_formulaParamInfo;

    bool m_allocate;    // expand, expandMacro 처리 동안에 실제 파라미터 메모리 블록을 할당할지 여부를 알려주는 임시 플래그

    VExecTracerPool m_tracerPools[2];

    static VStrList ms_builtinCustom;
    //static VStrList ms_builtinModel;
    static VStrList ms_builtinLayer;
    static VStrList ms_builtinNetwork;

protected:
    void m_setup(VModuleCore* srcCore, bool copyChildren);
    void m_setup(VDict moduleInfo);
    void m_setup(string sFormula, VDict paramInfo);

    void m_openGraph(int depth, VShape& shape, VDict shapeDict, bool trace);

    void m_createLayerGraph();
    void m_createNetworkGraph();

    /*
    void m_createModel();

    void m_createBertModel();
    void m_createTransformerEncoderModel();
    void m_createEfficientNetModel(char model_num);
    */

    VModule m_expandMacro(VShape& shape, VDict kwArgs);
    
    void m_resolve_repeat(int repeat);

    void m_appendChild(VModule child);

    VParameters m_getParameters();
    
    void m_setParamater(VTensorDict tensors, string mode);
    //void m_setParamaterKai(VTensorDict tensors);
    //void m_setParamaterTorch(VTensorDict tensors);

    void m_setMatchedParam(VDict pmset, VTensorDict tensors, string dstName, string srcName, string mode);

    VDict m_getSerializeInfo(bool bIncludeParam);
    VValue m_getParamInfo(VValue param, bool bIncludeParam);
    VDict m_getPmInfo(VDict pm, bool bIncludeParam);

    VValue m_loadParamInfo(VValue param);
    VDict m_loadPmInfo(VDict pm);

protected:
    void m_createModuleParam(int depth, VShape& shape, VDict shapeDict);
    void m_createFormulaParam(int depth, VShape& shape, VDict shapeDict);

    void m_createLinearParam(int depth, VShape& shape, VDict shapeDict, VDict param);
    void m_createDenseParam(int depth, VShape& shape, VDict shapeDict, VDict param);
    void m_createAddBiasParam(int depth, VShape& shape, VDict shapeDict, VDict param);
    void m_createConv2dParam(int depth, VShape& shape, VDict shapeDict, VDict param);
    void m_createConv2dDilatedParam(int depth, VShape& shape, VDict shapeDict, VDict param);
    void m_createBatchnormParam(int depth, VShape& shape, VDict shapeDict, VDict param);
    void m_createRnnParam(int depth, VShape& shape, VDict shapeDict, VDict param, string cell="");
    //void m_createLstmParam(int depth, VShape& shape, VDict shapeDict, VDict param);
    //void m_createGruParam(int depth, VShape& shape, VDict shapeDict, VDict param);
    void m_createActivateParam(int depth, VShape& shape, VDict shapeDict, VDict param, string func="");
    void m_createEmbedParam(int depth, VShape& shape, VDict shapeDict, VDict param);
    void m_createMultiHeadAttentionParam(int depth, VShape& shape, VDict shapeDict, VDict param);
    void m_createPoolParam(int depth, VShape& shape, VDict shapeDict, VDict param);
    void m_createUpsampleParam(int depth, VShape& shape, VDict shapeDict, VDict param);
    void m_createReshapeParam(int depth, VShape& shape, VDict shapeDict, VDict param);
    void m_createTransposeParam(int depth, VShape& shape, VDict shapeDict, VDict param);
    void m_createDropoutParam(int depth, VShape& shape, VDict shapeDict, VDict param);
    void m_createExtractParam(int depth, VShape& shape, VDict shapeDict, VDict param);
    void m_createLayernormParam(int depth, VShape& shape, VDict shapeDict, VDict param);
    void m_createFlattenParam(int depth, VShape& shape, VDict shapeDict, VDict param);
    void m_createPassParam(int depth, VShape& shape, VDict shapeDict, VDict param);
    void m_createGlobalAvgParam(int depth, VShape& shape, VDict shapeDict, VDict param);
    void m_createAdaptiveAvgParam(int depth, VShape& shape, VDict shapeDict, VDict param);
    void m_createConcatParam(int depth, VShape& shape, VDict shapeDict, VDict param);
    void m_createSelectNTopParam(int depth, VShape& shape, VDict shapeDict, VDict param);
    void m_createNormalNoiseParam(int depth, VShape& shape, VDict shapeDict, VDict param);
    void m_createUniformNoiseParam(int depth, VShape& shape, VDict shapeDict, VDict param);
    void m_createNormalRandomParam(int depth, VShape& shape, VDict shapeDict, VDict param);
    void m_createUniformRandomParam(int depth, VShape& shape, VDict shapeDict, VDict param);
    void m_createRoundParam(int depth, VShape& shape, VDict shapeDict, VDict param);
    void m_createCosineSimParam(int depth, VShape& shape, VDict shapeDict, VDict param);
    void m_createCodeConvParam(int depth, VShape& shape, VDict shapeDict, VDict param);

protected:
    static bool ms_inBuiltinCustomNames(string name);
    //static bool ms_inBuiltinModelNames(string name);
    static bool ms_inBuiltinLayerNames(string name);
    static bool ms_inBuiltinNetworkNames(string name);

public:
    void parallel_backprop(VTensor ygrad, VTensorList operands, VBackQueue* pQueue, VExecTracer tracer);
    //VTensor evaluateLayerOperation(VGraphCore* pGraphCore, VMath math, bool train, bool no_grad);   // VGraph 객체에서 VConsts::inOpLayerList()에 등록된 레이어에 한해레이어 모듈의 직접 처리를 의뢰함
    
    VList getDataIdx(int nDevice);
    string getFormula() { return m_formula; }
    VDict getFormulaParamInfo() { return m_formulaParamInfo; }
    bool isParamAllocated() { return m_allocate; }

protected:  // evaluateLayerOperation()가 호출할 레이어 종류별 처리 함수들 
    VTensorDict m_evaluate(VModuleCore* pStarter, VTensorDict xs, bool train, bool noGrad, int nDevice, VTensorDict& sideTerms, VExecTracer tracer);
    
    VGraph m_getGraph(VTensorDict& sideTerms);

    void m_invokeMatchingCallbacks(
        VModuleCore* pStarter, VTensorDict xs, VTensorDict ys, VTensorDict sideTerms, VParameters  params, 
        bool train, bool noGrad, int nDevice, bool bPre, VCbBackInfo cbInfo, VExecTracer tracer);

    int m_filterCheck(VCbItem item, bool train, int nDevice, bool bPre);
    int m_filterCheck(VCbItem item, int nDevice);

    void m_invokeCallback(
        VModuleCore* pStarter, VCbItem item, bool train, int nDevice, bool bPre,
        bool noGrad, VTensorDict xs, VTensorDict ys, VTensorDict sideTerms, VParameters  params, VExecTracer tracer);

    void m_splitDataIdx(int nDivisions);

    VShape m_get2dArg(string plainKey, string elemPrefix, string shapeName, int64 nDef1, int64 nDef2=-1);

    void m_getPaddingArg(VDict param, VShape& shape, VShape ksize, VShape stride);

    static float ms_getInitArg(string init_method, int64 in_width, int64 out_width, VDict args, float def=0.03f);

    //VTensor m_evalLayerConv(VGraphCore* pGraphCore, VMath math, bool train, bool no_grad);
    //VTensor m_getParam(VDict params, string key, VMath math, bool train, bool no_grad);
};
