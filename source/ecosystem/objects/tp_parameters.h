#pragma once

#include "../utils/tp_common.h"

class EParametersCore;
class EParameters {
public:
    EParameters();
    EParameters(ENN nn);
    EParameters(ENN nn, VHParameters hParameters);
    EParameters(const EParameters& src);
    EParameters(EParametersCore* core);
    virtual ~EParameters();
    EParameters& operator =(const EParameters& src);
    operator VHParameters();
    bool isValid();
    void close();
    ENN nn();
    EParametersCore* getCore();
    EParametersCore* cloneCore();
    int meNth();
    int meRefCnt();
    int handleNth();
    int handleRefCnt();
    EParametersCore* createApiClone();

protected:
    EParametersCore* m_core;

public:
    //void add(EParameters params);
    void zero_grad();
    void initWeights();

    VList weightList(ETensorDict& tensors);
    VList gradientList(ETensorDict& tensors);

    ETensorDict weightDict();
    ETensorDict gradientDict();
};
