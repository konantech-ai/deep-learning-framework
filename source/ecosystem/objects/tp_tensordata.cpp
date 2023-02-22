#include "../objects/tp_tensordata.h"
#include "../objects/tp_tensordata_core.h"
#include "../objects/tp_nn.h"
#include "../connect/tp_api_conn.h"
#include "../utils/tp_utils.h"
#include "../utils/tp_exception.h"

ETensorData::ETensorData() { m_core = NULL; }
ETensorData::ETensorData(ENN nn) { m_core = new ETensorDataCore(nn); }
ETensorData::ETensorData(const ETensorData& src) { m_core = src.m_core->clone(); }
ETensorData::ETensorData(ETensorDataCore* core) { m_core = core->clone(); }
ETensorData::~ETensorData() { m_core->destroy(); }
ETensorData& ETensorData::operator =(const ETensorData& src) {
    if (&src != this && m_core != src.m_core) {
        m_core->destroy();
        m_core = src.m_core->clone(); }
    return *this; }
bool ETensorData::isValid() { return m_core != NULL; }
void ETensorData::close() { if (this) m_core->destroy(); }
ENN ETensorData::nn() { return m_core ? m_core->m_nn : ENN(); }
ETensorDataCore* ETensorData::getCore() { return m_core; }
ETensorDataCore::ETensorDataCore(ENN nn) : VObjCore(VObjType::custom) { m_nn = nn; m_setup(); }
ETensorDataCore* ETensorData::createApiClone() { return m_core->clone(); }

//-----------------------------------------------------------------------------------------------------
// Capsule part

int ms_alloc_count = 0;
int ms_free_count = 0;
int64 ms_alloc_size = 0;
int64 ms_free_size = 0;

ETensorData::ETensorData(ENN nn, int64 size) {
    m_core = new ETensorDataCore(nn);
    m_core->m_byteSize = size;
    m_core->m_pData = malloc(size);

    if (m_core->m_pData == NULL) TP_THROW2(VERR_HOSTMEM_ALLOC_FAILURE, to_string(size));
}

void ETensorData::reset() {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    memset(m_core->m_pData, 0, m_core->m_byteSize);
}

void ETensorData::fillFloatData(VShape shape, VList values) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    m_core->m_fillFloatData(shape, 0, values);
}

void ETensorData::fillIntData(VShape shape, VList values) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    m_core->m_fillIntData(shape, 0, values);
}

void ETensorData::fillData(VShape shape, float value) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    m_core->m_fillData(shape, value);
}

void ETensorData::fillData(VShape shape, int value) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    m_core->m_fillData(shape, value);
}

void ETensorData::copyFrom(void* pData) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    memcpy(m_core->m_pData, pData, m_core->m_byteSize);
}

void ETensorData::readFrom(TpStreamIn* fin) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    fin->load_data(m_core->m_pData, m_core->m_byteSize);
}

void ETensorData::readFrom(FILE* fin) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    TpStreamIn::load_data(fin, m_core->m_pData, m_core->m_byteSize);
}

void ETensorData::downloadFrom(VHTensor hTensor) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    nn().getApiConn()->Tensor_downloadData(hTensor, m_core->m_pData, m_core->m_byteSize, __FILE__, __LINE__);
}

void ETensorData::downloadFrom(ENN nn, VHTensor hTensor) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    nn.getApiConn()->Tensor_downloadData(hTensor, m_core->m_pData, m_core->m_byteSize, __FILE__, __LINE__);
}

int64 ETensorData::byteSize() {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    return m_core->m_byteSize;
}

void* ETensorData::void_ptr() const {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    return m_core->m_pData;
}

int* ETensorData::int_ptr() const {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    return (int*)m_core->m_pData;
}

int64* ETensorData::int64_ptr() const {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    return (int64*)m_core->m_pData;
}

float* ETensorData::float_ptr() const {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    return (float*)m_core->m_pData;
}

unsigned char* ETensorData::uchar_ptr() const {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    return (unsigned char*)m_core->m_pData;
}

void ETensorData::save(TpStreamOut* gout) {
    if (m_core == NULL) {
        gout->save_int64(0);
        return;
        //TP_THROW(VERR_INVALID_CORE); // 추후 throw 필요한 경우 확인되면 bThrow 인자 두어 옵션화
    }

    gout->save_int64(m_core->m_byteSize);
    gout->save_data(m_core->m_pData, m_core->m_byteSize);
}

//-----------------------------------------------------------------------------------------------------
// Core part

void ETensorDataCore::m_setup() {
    m_byteSize = 0;
    m_pData = NULL;
}

ETensorDataCore::~ETensorDataCore() {
    if (m_pData) {
        free(m_pData);

        m_pData = NULL;
        m_byteSize = 0;
    }
}

void ETensorDataCore::m_fillFloatData(VShape shape, int64 nth, VList values) {
    if (shape.size() <= 0)  TP_THROW(VERR_INVALID_ARGUMENT);

    int64 size = shape[0];

    if (values.size() != size) TP_THROW(VERR_INVALID_ARGUMENT);

    if (shape.size() == 1) {
        float* pData = ((float*)m_pData) + nth;
        for (int64 n = 0; n < size; n++) {
            pData[n] = values[n];
        }
    }
    else {
        VShape tshape = shape.remove_head();
        int64 child_size = tshape.total_size();

        for (int64 n = 0; n < size; n++) {
            if (!values[n].is_list()) TP_THROW(VERR_INVALID_ARGUMENT);
            VList sub_values = values[n];
            m_fillFloatData(tshape, nth, sub_values);
            nth += child_size;
        }
    }
}

void ETensorDataCore::m_fillIntData(VShape shape, int64 nth, VList values) {
    if (shape.size() <= 0)  TP_THROW(VERR_INVALID_ARGUMENT);

    int64 size = shape[0];

    if (values.size() != size) TP_THROW(VERR_INVALID_ARGUMENT);

    if (shape.size() == 1) {
        int* pData = ((int*)m_pData) + nth;
        for (int64 n = 0; n < size; n++) {
            pData[n] = values[n];
        }
    }
    else {
        VShape tshape = shape.remove_head();
        int64 child_size = tshape.total_size();

        for (int64 n = 0; n < size; n++) {
            if (!values[n].is_list()) TP_THROW(VERR_INVALID_ARGUMENT);
            VList sub_values = values[n];
            m_fillIntData(tshape, nth, sub_values);
            nth += child_size;
        }
    }
}

void ETensorDataCore::m_fillData(VShape shape, float value) {
    if (shape.size() <= 0)  TP_THROW(VERR_INVALID_ARGUMENT);

    int64 size = shape[0];

    float* pData = (float*)m_pData;

    for (int64 n = 0; n < size; n++) {
        pData[n] = value;
    }
}

void ETensorDataCore::m_fillData(VShape shape, int value) {
    if (shape.size() <= 0)  TP_THROW(VERR_INVALID_ARGUMENT);

    int64 size = shape[0];

    int* pData = (int*)m_pData;

    for (int64 n = 0; n < size; n++) {
        pData[n] = value;
    }
}

