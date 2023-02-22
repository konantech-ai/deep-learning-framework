#pragma once

#include "../api/vcommon.h"

class VConsts {
public:
    static string getLayerExpression(string sLayerName);
    static string getLossExpression(string sLossName);
    static string getMetricExpression(string sInfName);

    //static bool inOpLayerList(string sLayername);

    static VGraphOpCode convToOpCode(string opcode);

    static OptAlgorithm getOptAlgorithm(string sBuiltin);

    static ActFunc getActFunc(string sActFunc);

    static bool needXForBackwardgetActFunc(ActFunc func);
};
