#pragma once

#include "../utils/tp_common.h"

class EScalarCore;
class EScalar {
public:
    EScalar();
    EScalar(ENN nn);
    EScalar(const EScalar& src);
    EScalar(EScalarCore* core);
    virtual ~EScalar();
    EScalar& operator =(const EScalar& src);
    bool isValid();
    void close();
    ENN nn();
    EScalarCore* getCore();
    EScalarCore* createApiClone();
protected:
    EScalarCore* m_core;
public:

public:
    EScalar(ENN nn, ETensor tensor, float value);
    operator float() const;
};
