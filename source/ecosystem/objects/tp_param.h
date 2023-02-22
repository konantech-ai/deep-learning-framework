#pragma once

#include "../utils/tp_common.h"
#include "../objects/tp_tensor.h"

#ifdef NOT_DEPRECIATED
class Param : public ETensor {
public:
    Param(ENN nn, ETensor init);
    virtual ~Param();
};
#endif