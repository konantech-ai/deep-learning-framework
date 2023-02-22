#pragma once

#include "../utils/tp_common.h"
#include "../objects/tp_tensordata.h"
#include "../objects/tp_nn.h"

class ETensorCore : public EcoObjCore {
protected:
    friend class ETensor;
protected:
    ETensorCore(ENN nn, VHTensor hTensor);
    ~ETensorCore();
    ETensorCore* clone() { return (ETensorCore*)clone_core(); }
    void m_setup();
    void m_delete();

protected:
    ENN m_nn;
    //VHTensor m_hTensor;

protected:
    void m_createData(void* pData);
    void m_createData(FILE* fin, VDataType loadType);
    void m_downloadData(VHTensor hTensor);
    void m_downloadData(ENN nn, VHTensor hTensor);
    void m_copyData(void* pData);

    void m_reset();

    int m_fetchIntScalar();
    int64 m_fetchInt64Scalar();
    float m_fetchFloatScalar();

    void m_to_type(ETensor src, string option);
    void m_fetchIdxRows(ETensor src, int64* pnMap, int64 size);
    void m_fetchIdxRows(ETensor src, int* pnMap, int64 size);
    void m_copy_into_row(int64 nthRow, ETensor src);

    void m_argmax(ETensor opnd, int64 axis);
    void m_sum(ETensor opnd);
    void m_mean(ETensor opnd);
    void m_abs(ETensor opnd);
    void m_sigmoid(ETensor opnd);
    void m_square(ETensor opnd);
    void m_transpose(ETensor opnd);
    void m_max(ETensor opnd1, ETensor opnd2);

    void m_mult(ETensor opnd1, ETensor opnd2);
    void m_div(ETensor opnd1, ETensor opnd2);
    void m_add(ETensor opnd1, ETensor opnd2);
    void m_subtract(ETensor opnd1, ETensor opnd2);

protected:
    bool m_needToClose;
    bool m_needToUpload;

    VShape m_shape;
    VDataType m_type;
    int m_nDevice;

    ETensorData m_data;

    string m_debugName;
};
