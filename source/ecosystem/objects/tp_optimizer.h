#pragma once

#include "../utils/tp_common.h"

class EOptimizerCore;

class EOptimizer {
   
public:
    EOptimizer();
    EOptimizer(ENN nn);
    EOptimizer(ENN nn, VHOptimizer hOptimizer);
    EOptimizer(const EOptimizer& src);
    EOptimizer(EOptimizerCore* core);
    virtual ~EOptimizer();
    EOptimizer& operator =(const EOptimizer& src);
    operator VHOptimizer();
    bool isValid();
    void close();
    ENN nn();
    EOptimizerCore* getCore();
    EOptimizerCore* cloneCore();
    int meNth();
    int meRefCnt();
    int handleNth();
    int handleRefCnt();
    EOptimizerCore* createApiClone();
protected:
    EOptimizerCore* m_core;

public:
    void setup(string sBuiltin, VDict kwArgs, EParameters params);

public:
    void setOption(VDict kwArgs);

    void zero_grad();
    void step();

protected:
    //static EOptimizer ms_createOptimizer(string sBuiltin, EParameters params, VDict kwArgs);

};
