#include <cuda_runtime.h>

#include "../api/vconst.h"
#include "../local_objects/vgraph.h"
#include "../local_objects/vgraph_core.h"
#include "../api_objects/vmodule_core.h"
#include "../api_objects/vtensor.h"
#include "../api_objects/vmodule.h"
#include "../local_objects/vdevicemanager.h"
#include "../support/vmath.h"
#include "../utils/vutils.h"

VGraph::VGraph() {
    m_core = NULL;
}

VGraph::VGraph(VSession session, string sBuiltin, VDict kwArgs) {
    m_core = new VGraphCore(session, sBuiltin, kwArgs);
}

VGraph::VGraph(const VGraph& src) {
    m_core = src.m_core->clone();
}

VGraph::VGraph(VGraphCore* core) {
    m_core = core->clone();
}

VGraph::~VGraph() {
    m_core->destroy();
}

VGraph& VGraph::operator =(const VGraph& src) {
    if (&src != this && m_core != src.m_core) {
        m_core->destroy();
        m_core = src.m_core->clone();
    }
    return *this;
}

VGraphCore* VGraph::getClone() {
    return (VGraphCore*)m_core->clone_core();
}

VGraphCore* VGraph::getCore() {
    return m_core;
}

void VGraph::destroyCore() {
    if (m_core->getRefCnt() > 1) m_core->destroy();
    else {
        m_core->destroy();
        m_core = NULL;
    }
}

VSession VGraph::session() const {
    return m_core->m_session;
}

bool VGraph::isValid() {
    return m_core != NULL;
}

int VGraph::getRefCnt() {
    return m_core->getRefCnt();
}

int VGraph::getNth() {
    return m_core->getNth();
}

VGraphCore::VGraphCore(VSession session, string sBuiltin, VDict kwArgs) : VObjCore(VObjType::Graph) {
    m_sBuiltin = vutils.tolower(sBuiltin);
    m_propDict = kwArgs;
    m_session = session,
        m_setup();
}

int VGraphCore::ms_nNextTouchVersion = 0;

VGraph::VGraph(VSession session, VModuleCore* pModuleCore, GraphInit init, VDeviceManager devman, string sName, VList params, VDict kwArgs) {
    m_core = new VGraphCore(session, sName, kwArgs);
    m_core->m_nTouchVersion = m_core->ms_nNextTouchVersion++;
    m_core->m_deviceMan = devman;
    m_core->m_pOptimizer = 0;

    string sExp;
    string reduction;
    
    switch (init) {
    case GraphInit::layer:
        /*if (VConsts::inOpLayerList(sName)) {
            m_core->m_rootNode = NULL;
            m_core->m_pModuleCore = pModuleCore;
            return;
        }
        else */
        sExp = pModuleCore->getFormula();
        if (sExp == "") {
            sExp = VConsts::getLayerExpression(sName);
        }
        else {
            //params = VList{ pModuleCore->getFormulaParamInfo() };
        }
        break;
    case GraphInit::loss:
        if (sName == "custom") {   // custom 방식을 GraphInit::term 이용하는 방식으로 수정함. 기존 에제 가운데 걸리는 것 있으면 적절히 보완할 것
            VP_THROW(VERR_GRAPH_INIT);
            sExp = kwArgs["exp"];
        }
        sExp = VConsts::getLossExpression(sName);
        reduction = (string)vutils.seek_dict(kwArgs, "reduction", "sum");
        if (reduction != "") sExp = reduction + "(" + sExp + ")";
        break;
    case GraphInit::metric:
        if (sName == "custom") {   // custom 방식을 GraphInit::term 이용하는 방ㄹ식으로 수정함. 기존 에제 가운데 걸리는 것 있으면 적절히 보완할 것
            VP_THROW(VERR_GRAPH_INIT);
            sExp = kwArgs["exp"];
        }
        sExp = VConsts::getMetricExpression(sName);
        reduction = (string)vutils.seek_dict(kwArgs, "reduction", "sum");
        if (reduction != "") sExp = reduction + "(" + sExp + ")";
        break;
    case GraphInit::term:
        m_core->m_sBuiltin = "term";
        sExp = sName;  // 매개변수값 전달에 약간의 편법 사용중
        break;
    default:
        VP_THROW(VERR_GRAPH_INIT);
        break;
    }

    if (pModuleCore) m_core->m_pmAllocaed = pModuleCore->isParamAllocated();
    else {
        m_core->m_pmAllocaed = true;
    }

	int pos = 0;
    m_core->m_rootNode = VGraphNode(session, sExp, params, pos, devman, m_core);
}

VGraph::VGraph(VSession session, string sBuiltin, VDeviceManager devman, VDict kwArgs) {
    m_core = new VGraphCore(session, "", kwArgs);
    m_core->m_nTouchVersion = m_core->ms_nNextTouchVersion++;
    m_core->m_deviceMan = devman;

    m_core->m_rootNode = NULL;
    m_core->m_sNetName = sBuiltin;

    if (sBuiltin == "residual") {
        string sExp = VConsts::getLayerExpression("residual_net");
        int pos = 0;
        m_core->m_rootNode = VGraphNode(session, sExp, {}, pos, devman, m_core);
    }
    else if (sBuiltin == "parallel") {
        string sExp = VConsts::getLayerExpression("parallel_net");
        int pos = 0;
        m_core->m_rootNode = VGraphNode(session, sExp, {}, pos, devman, m_core);
    }
    else if (sBuiltin == "add") {
        string sExp = VConsts::getLayerExpression("add_net");
        int pos = 0;
        m_core->m_rootNode = VGraphNode(session, sExp, {}, pos, devman, m_core);
    }
    else if (sBuiltin == "stack") {
        string sExp = VConsts::getLayerExpression("stack_net");
        int pos = 0;
        m_core->m_rootNode = VGraphNode(session, sExp, {}, pos, devman, m_core);
    }
    else if (sBuiltin == "squeezeexcitation") {
        string sExp = VConsts::getLayerExpression("se_net");
        int pos = 0;
        m_core->m_rootNode = VGraphNode(session, sExp, {}, pos, devman, m_core);
    }
    //m_core->m_pNetModuleCore = pModuleCore;
}

/*
VGraph::VGraph(string sLayerName, VList params, VDict kwArgs) {
    m_core = new VGraphCore("", kwArgs);
    m_core->m_nTouchVersion = m_core->ms_nNextTouchVersion++;
    
    string sLayerExp = VConsts::getLayerExpression(sLayerName);

	int pos = 0;
	m_core->m_rootNode = VGraphNode(sLayerExp, params, pos);
}

VGraph::VGraph(string sLossName, VDict kwArgs) {
    m_core = new VGraphCore("", kwArgs);
    m_core->m_nTouchVersion = m_core->ms_nNextTouchVersion++;

    string sLossExp = VConsts::getLossExpression(sLossName);
    string reduction = vutils.seek_dict(kwArgs, "reduction", "mean");

    if (reduction != "") sLossExp = reduction + "(" + sLossExp + ")";

    int pos = 0;
    m_core->m_rootNode = VGraphNode(sLossExp, {}, pos);
}
*/

VGraph::VGraph(GraphInit init, VDeviceManager devman, const VGraph& src) {
    m_core = new VGraphCore(src.session(), "", VDict());
    m_core->m_nTouchVersion = src.m_core->m_nTouchVersion;
    m_core->m_sNetName = src.m_core->m_sNetName;
    m_core->m_propDict = vutils.copy(src.m_core->m_propDict);
    //m_core->m_pModuleCore = src.m_core->m_pModuleCore;
    m_core->m_deviceMan = devman;

    if (init != GraphInit::deep_copy) VP_THROW(VERR_GRAPH_INIT);

    if (src.m_core->m_rootNode.isValid()) {
        m_core->m_rootNode = VGraphNode(true, src.m_core->m_rootNode, devman, m_core);
    }
}

int VGraph::getDevice() {
    return m_core->m_device;
}

VValue VGraph::getOption(string key) {
    return m_core->m_propDict[key];
}

void VGraph::setOption(string key, VValue value) {
    m_core->m_propDict[key] = value;
}

void VGraph::setSideTerms(VTensorDict sideTerms) {
    m_core->m_sideTerms = sideTerms;
}

VTensorDict VGraph::evaluateGraph(VTensorDict xs, bool train, bool noGrad, int device, VCbBackInfo cbInfo, VExecTracer tracer) {
    m_core->m_xs = xs;
    m_core->m_ys.clear();
    if (0) {
        printf("\nVGraph::evaluateGraph1(train=%d)\n\n", train);
    }
    m_core->m_train = train;
    m_core->m_noGrad = noGrad;
    m_core->m_device = device;

    VDict args; // 내부 정보 수집 및 전달용이어서 탑레벨에서는 결과값을 이용할 필요가 없음

    if (m_core->m_rootNode.isValid()) {
        m_core->m_ys["#"] = m_core->m_rootNode.evaluate(args, cbInfo, tracer);
    }
    /*
    else if (m_core->m_pModuleCore) {
        m_core->m_ys["#"] = m_core->m_pModuleCore->evaluateLayerOperation(m_core, m_core->m_math, train, noGrad);
    }
    */
    else {
        VP_THROW(VERR_NOT_IMPLEMENTED_YET);
        /*if (m_core->m_sNetName == "sequential") {
        VTensorDict vars = xs;
        for (auto& it : m_core->m_netChildren) {
            vars = it.evaluate(vars, train, noGrad, device);
        }
        m_core->m_ys = vars;
        */
    }

    VTensorDict result = m_core->m_ys;

    m_core->m_xs = VTensorDict();
    m_core->m_ys = VTensorDict();

    return result;
}

void VGraph::evaluateGraph(VTensorDict xs, VTensorDict* pCustomTerms, string sName, VGraphDict graphs, bool train, bool noGrad, int device, VCbBackInfo cbInfo, VExecTracer tracer) {
    m_core->m_xs = xs;
    m_core->m_pCustomTerms = pCustomTerms;
    
    if (0) printf("\nVGraph::evaluateGraph2(train=%d)\n\n", train);
    m_core->m_train = train;
    m_core->m_noGrad = noGrad;
    m_core->m_device = device;
    m_core->m_graphs = graphs;

    VDict args; // 내부 정보 수집 및 전달용이어서 탑레벨에서는 결과값을 이용할 필요가 없음

    if (m_core->m_rootNode.isValid()) {
        (*m_core->m_pCustomTerms)[sName] = m_core->m_rootNode.evaluate(args, cbInfo, tracer);
    }
}

//--------------------------------------------------------------------------------------------------

VTensor VGraphCore::seekInput(string varName) {
    if (m_xs.find(varName) != m_xs.end()) return m_xs[varName]; 

    if (varName[0] == '#' && varName.size() > 1) {
        varName = varName.substr(1);
        if (m_xs.find(varName) != m_xs.end()) return m_xs[varName];
    }

    if (varName == "x" && m_xs.find("#") != m_xs.end()) return m_xs["#"];
    if (varName == "y" && m_xs.find("#y:#") != m_xs.end()) return m_xs["#y:#"];
    if (varName == "pred" && m_xs.find("#pred:#") != m_xs.end()) return m_xs["#pred:#"];
    if (m_xs.find("#y:" + varName) != m_xs.end()) return m_xs["#y:" + varName];
    if (m_xs.find("#pred:" + varName) != m_xs.end()) return m_xs["#pred:" + varName];

    if (1) {
        printf("varName = %s is not found\n", varName.c_str());
        for (auto& it : m_xs) {
            printf("m_xs[%s] = #%d\n", it.first.c_str(), it.second.getNth());
        }
    }

    VP_THROW(VERR_NOT_IMPLEMENTED_YET);
    
}

VTensor VGraphCore::m_seekTerm(string varName, VCbBackInfo cbInfo, VExecTracer tracer) {
    if (m_pCustomTerms->find(varName) != m_pCustomTerms->end()) {
        VTensor tensor = (*m_pCustomTerms)[varName];
        if (!tensor.isValid()) {
            printf("invalid term '%s' used\n", varName.c_str());
            VP_THROW(VERR_INVALID_TENSOR); // 재귀적 호출에 해당하므로 예외 처리 필요
        }
        return tensor;
    }
    else if (m_graphs.find(varName) != m_graphs.end()) {
        (*m_pCustomTerms)[varName] = VTensor();	// 재귀적 호출 탐지를 위해 일단 더미값을 설정해둔다.
        m_graphs[varName].evaluateGraph(m_xs, m_pCustomTerms, varName, m_graphs, m_train, m_noGrad, m_device, cbInfo, tracer);
        return (*m_pCustomTerms)[varName];
    }
    else {
        printf("BP1: lost varName in expression: %s\n", varName.c_str());
        VP_THROW1(VERR_INVALID_DICT_KEY, varName);
    }
}

void VGraphCore::m_keepOutput(string varName, VTensor tensor) {
    m_ys[varName] = tensor;
}

void VGraphCore::m_setup() {
    m_train = false;
    //m_pModuleCore = NULL;
}

VTensor VGraphCore::optimizerPreproc(VTensor param, VDict pmset, VExecTracer tracer) {
    return m_pOptimizer.preproc(param, pmset, tracer);
}

