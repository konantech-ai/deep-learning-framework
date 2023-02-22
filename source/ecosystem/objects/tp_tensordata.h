#pragma once

#include "../utils/tp_common.h"

class TpStreamIn;
class TpStreamOut;

class ETensorDataCore;

class ETensorData {
public:
    ETensorData();
    ETensorData(ENN nn);
    ETensorData(const ETensorData& src);
    ETensorData(ETensorDataCore* core);
    virtual ~ETensorData();
    ETensorData& operator =(const ETensorData& src);
    bool isValid();
    void close();
    ENN nn();
    ETensorDataCore* getCore();
    ETensorDataCore* createApiClone();

protected:
    ETensorDataCore* m_core;

public:
    ETensorData(ENN nn, int64 size);

    void reset();
    void copyFrom(void* pData);
    void readFrom(TpStreamIn* fin);
    void readFrom(FILE* fin);
    void downloadFrom(VHTensor hTensor);
    void downloadFrom(ENN nn, VHTensor hTensor);
    void fillFloatData(VShape shape, VList values);
    void fillIntData(VShape shape, VList values);
    void fillData(VShape shape, float value);
    void fillData(VShape shape, int value);

    int64 byteSize();

    void* void_ptr() const;
    int* int_ptr() const;
    int64* int64_ptr() const;
    float* float_ptr() const;
    unsigned char* uchar_ptr() const;

    void save(TpStreamOut* fout);

protected:
    void m_close();
};
