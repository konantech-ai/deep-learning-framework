#pragma once

#include "../utils/tp_common.h"

#ifdef NOT_DEPRECIATED
class Math {
public:
    static ETensor evaluate(string expression, VList args);

    static ETensor linspace(float from, float to, int count, VDict kwArgs);
    static ETensor sin(ETensor);
    static ETensor randh(VShape shape, VDict kwArgs);
};
#endif