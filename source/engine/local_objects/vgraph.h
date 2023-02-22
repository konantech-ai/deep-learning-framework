#pragma once

#include "../api/vcommon.h"
#include "../api_objects/vsession.h"
#include "../local_objects/vexectracer.h"

class VGraph;
class VTensor;
class VModuleCore;
class VDeviceManager;
class VCbBackInfo;

typedef map<string, VGraph> VGraphDict;

enum class GraphInit { layer, loss, metric, deep_copy, term };

class VGraphCore;

class VGraph {
public:
    VGraph();
    VGraph(VSession session, string sBuiltin, VDict kwArgs = {});
    VGraph(const VGraph& src);
    VGraph(VGraphCore* core);
    virtual ~VGraph();
    VGraph& operator =(const VGraph& src);
    VGraphCore* getClone();
    VGraphCore* getCore();
    void destroyCore();
    VSession session() const;
    bool isValid();
    int getRefCnt();
    int getNth();
protected:
    VGraphCore* m_core;

public:
    VGraph(VSession session, VModuleCore* pModuleCore, GraphInit init, VDeviceManager devman, string sLayerName, VList params, VDict kwArgs);
    VGraph(VSession session, string sBuiltin, VDeviceManager devman, VDict kwArgs);
    VGraph(GraphInit init, VDeviceManager devman, const VGraph& src);

    VTensorDict evaluateGraph(VTensorDict xs, bool train, bool noGrad, int device, VCbBackInfo cbInfo, VExecTracer tracer);
    void evaluateGraph(VTensorDict xs, VTensorDict* pCustomTerms, string sName, VGraphDict graphs, bool train, bool noGrad, int device, VCbBackInfo cbInfo, VExecTracer tracer);

    int getDevice();

    VValue getOption(string key);
    void setOption(string key, VValue value);
    void setSideTerms(VTensorDict sideTerms);

};
