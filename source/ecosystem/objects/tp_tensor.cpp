#include "../objects/tp_tensor.h"
#include "../objects/tp_tensor_core.h"
#include "../objects/tp_scalar.h"
#include "../objects/tp_nn.h"
#include "../connect/tp_api_conn.h"
#include "../utils/tp_utils.h"
#include "../utils/tp_exception.h"
#include "../utils/tp_cuda.h"

//-----------------------------------------------------------------------------------------------------
// Capsule part

static std::random_device ms_rd{};
//static std::mt19937 ms_randGen{ ms_rd() };
static std::mt19937 ms_randGen{ 1234 };

ETensor::ETensor() { m_core = NULL; }
ETensor::ETensor(ENN nn) { m_core = new ETensorCore(nn, 0); }
ETensor::ETensor(ENN nn, VHTensor hTensor) { m_core = new ETensorCore(nn, hTensor); }
ETensor::ETensor(const ETensor& src) { m_core = src.m_core->clone(); }
ETensor::ETensor(ETensorCore* core) { m_core = core->clone(); }
ETensor::~ETensor() { m_core->destroy(); }

ETensor& ETensor::operator =(const ETensor& src) {
    if (&src != this && m_core != src.m_core) {
        m_core->destroy();
        m_core = src.m_core->clone();
    }
    return *this;
}

ETensor::operator VHTensor() { return m_core->m_hEngineHandle; }
bool ETensor::isValid() { return m_core != NULL; }
void ETensor::close() { if (this) m_core->destroy(); }
ENN ETensor::nn() { return m_core ? m_core->m_nn : ENN(); }
ETensorCore* ETensor::getCore() { return m_core; }
ETensorCore* ETensor::cloneCore() { return (ETensorCore*) m_core->clone(); }
int ETensor::meNth() { return m_core->getNth(); }
int ETensor::meRefCnt() { return m_core->getRefCnt(); }

ETensorCore::ETensorCore(ENN nn, VHTensor hTensor) : EcoObjCore(VObjType::custom) {
    m_nn = nn;
    m_hEngineHandle = hTensor;
    m_setup();
}
ETensorCore::~ETensorCore() {
    m_delete();
    m_nn.getApiConn()->Tensor_close(m_hEngineHandle, __FILE__, __LINE__);
}
ETensorCore* ETensor::createApiClone() { return m_core->clone(); }

/*
ETensor::ETensor(VShape shape, VDataType type) {
    m_core = new ETensorCore(ENN(), 0);

    m_core->m_shape = shape;
    m_core->m_type = type;

    m_core->m_createData(NULL);
}
*/

ETensor::ETensor(ENN nn, VShape shape, VDataType type, void* pData) {
    m_core = new ETensorCore(nn, 0);

    m_core->m_shape = shape;
    m_core->m_type = type;
    m_core->m_nDevice = -1;

    m_core->m_createData(pData);
}

ETensor::ETensor(ENN nn, VShape shape, VDataType dataType, VDataType loadType, FILE* fin) {
    m_core = new ETensorCore(nn, 0);

    m_core->m_shape = shape;
    m_core->m_type = dataType;
    m_core->m_nDevice = -1;

    m_core->m_createData(fin, loadType);
}

ETensor::ETensor(ENN nn, VShape shape, VDataType type, VList values) {
    m_core = new ETensorCore(nn, 0);

    m_core->m_shape = shape;
    m_core->m_type = type;
    m_core->m_nDevice = -1;

    m_core->m_createData(NULL);

    if (type == VDataType::float32) {
        m_core->m_data.fillFloatData(shape, values);
    }
    else if (type == VDataType::int32) {
        m_core->m_data.fillIntData(shape, values);
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }
}

ETensor::ETensor(ENN nn, VShape shape, VDataType type, int device) {
    m_core = new ETensorCore(nn, 0);

    m_core->m_shape = shape;
    m_core->m_type = type;
    m_core->m_nDevice = device;

    m_core->m_createData(NULL);
}

ETensor::ETensor(ENN nn, VShape shape, VDataType type, string initMethod) {
    m_core = new ETensorCore(nn, 0);

    m_core->m_shape = shape;
    m_core->m_type = type;
    m_core->m_nDevice = -1;

    m_core->m_createData(NULL);

    m_core->m_createData(NULL);

    if (initMethod == "zeros") {
        if (type == VDataType::float32) {
            m_core->m_data.fillData(shape, 0.0f);
        }
        else if (type == VDataType::int32) {
            m_core->m_data.fillIntData(shape, 0);
        }
        else {
            TP_THROW(VERR_NOT_IMPLEMENTED_YET);
        }
    }
    else if (initMethod == "ones") {
        if (type == VDataType::float32) {
            m_core->m_data.fillData(shape, 1.0f);
        }
        else if (type == VDataType::int32) {
            m_core->m_data.fillData(shape, 1);
        }
        else {
            TP_THROW(VERR_NOT_IMPLEMENTED_YET);
        }
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }
}

ETensor::ETensor(ETensor src, VShape shape) {
    m_core = new ETensorCore(src.nn(), 0);

    m_core->m_shape = shape;
    m_core->m_type = src.m_core->m_type;
    m_core->m_nDevice = src.m_core->m_nDevice;

    m_core->m_data = src.m_core->m_data;
}

ETensor::ETensor(ENN nn, VHTensor hTensor, bool needToClose, bool needToUpload) {
    m_core = new ETensorCore(nn, hTensor);
    
    m_core->m_needToClose = needToClose;
    m_core->m_needToUpload = needToUpload;

    nn.getApiConn()->Tensor_getFeature(hTensor, &m_core->m_shape, &m_core->m_type, &m_core->m_nDevice, __FILE__, __LINE__);
}

VShape ETensor::shape() const {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    return m_core->m_shape;
}

int ETensor::device() {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    return m_core->m_nDevice;
}

VDataType ETensor::type() const {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    return m_core->m_type;
}

string ETensor::type_desc() const {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    return TpUtils::to_string(m_core->m_type);
}

bool ETensor::hasNoData() {
    if (m_core == NULL) return true;
    return !m_core->m_data.isValid();
}

void ETensor::allocData() {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    m_core->m_createData(NULL);
}

void ETensor::downloadData() {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    m_core->m_downloadData(m_core->m_hEngineHandle);
}

void ETensor::downloadData(ENN nn) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    m_core->m_downloadData(nn, m_core->m_hEngineHandle);
}

void ETensor::copyData(VShape shape, VDataType type, void* pData) {
    if (m_core == NULL) TP_THROW(VERR_UNDEFINED);
    if (m_core->m_shape != shape) TP_THROW(VERR_UNDEFINED);
    if (m_core->m_type != type) TP_THROW(VERR_UNDEFINED);
    m_core->m_copyData(pData);
}

void* ETensor::void_ptr() {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    if (hasNoData()) downloadData();
    return m_core->m_data.void_ptr();
}

int* ETensor::int_ptr() {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    if (type() != VDataType::int32) TP_THROW(VERR_TENSOR_DATATYPE);
    if (hasNoData()) downloadData();
    return m_core->m_data.int_ptr();
}

int64* ETensor::int64_ptr() {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    if (type() != VDataType::int64) TP_THROW(VERR_TENSOR_DATATYPE);
    if (hasNoData()) downloadData();
    return m_core->m_data.int64_ptr();
}

float* ETensor::float_ptr() {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    if (type() != VDataType::float32) TP_THROW(VERR_TENSOR_DATATYPE);
    if (hasNoData()) {
        downloadData();
    }
    return m_core->m_data.float_ptr();
}

unsigned char* ETensor::uchar_ptr() {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    if (type() != VDataType::uint8) {
        TP_THROW(VERR_TENSOR_DATATYPE);
    }
    if (hasNoData()) downloadData();
    return m_core->m_data.uchar_ptr();
}

unsigned char* ETensor::bool_ptr() {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    if (type() != VDataType::bool8) {
        TP_THROW(VERR_TENSOR_DATATYPE);
    }
    if (hasNoData()) downloadData();
    return m_core->m_data.uchar_ptr();  // 부울 값은 내부적으로 1byte 저장
}

void ETensor::reset() {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    m_core->m_reset();
}

void ETensor::setElement(int64 pos, VValue value) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    VShape xshape = shape();

    if (type() == VDataType::float32) {
        float* pElem = float_ptr() + pos;
        float fValue = value;

        memcpy(pElem, &fValue, sizeof(float));
    }
    else if (type() == VDataType::int32) {
        int* pElem = int_ptr() + pos;
        int nValue = value;

        memcpy(pElem, &nValue, sizeof(int));
    }
    else if (type() == VDataType::int64) {
        int64* pElem = int64_ptr() + pos;
        int64 nValue = value;

        memcpy(pElem, &nValue, sizeof(int64));
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }

}

void ETensor::setElement(VList pos, VValue value) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    int64 index = 0;
    VShape xshape = shape();

    if (pos.size() != xshape.size()) TP_THROW(VERR_SIZE_TENSOR);

    for (int n = 0; n < pos.size(); n++) {
        int64 npos = pos[n];
        if (npos < 0 || npos >= xshape[n]) TP_THROW(VERR_OUT_OF_RANGE);
        index = index * xshape[n] + npos;
    }

    setElement(index, value);
}

int64 m_setElement(int64 nth, VList index, VDataType dataType, char* pDst, int64& restSize, char*& pSrc, VShape shape) {
    if (nth >= index.size()) {
        return (shape.size() == 0) ? 1 : shape.valid_size();
    }

    VShape tshape = shape.remove_head();
    int64 tailSize = TpUtils::byte_size(dataType) * ((tshape.size() == 0) ? 1 : tshape.valid_size());

    if(index[nth].is_int64()) {
        if (shape[0] == 1) {
            return m_setElement(nth+1, index, dataType, pDst, restSize, pSrc, tshape);
        }
        else {
            pDst = pDst + (int64)index[nth] * tailSize;
            int64 blockSize = m_setElement(nth + 1, index, dataType, pDst, restSize,  pSrc, tshape);
            if (blockSize == 0) return 0;
            memcpy(pDst, pSrc, tailSize);
            restSize -= tailSize;
            pSrc += tailSize;
            return 0;
        }
    }
    else if (index[nth].is_list()) {
        VList triple = index[nth];

        int64 from = triple[0];
        int64 stop = triple[1];
        int64 step = triple[2];

        if (from < 0) from += shape[0];
        if (stop < 0) stop += shape[0];

        if (from == 0 && stop == shape[0] && step == 1) {
            int64 blockSize = m_setElement(nth + 1, index, dataType, pDst, restSize, pSrc, tshape);
            if (blockSize > 0) {
                return blockSize * shape[0];
            }
            from++;    // 시험삼아 첫 원소 시도했는데 내부에서 처리 완료, 따라서 첫 원소 스킵
        }

        pDst += from * tailSize;

        for (int64 n = from; (step > 0) ? (n < stop) : (n > stop); n += step) {
            int64 blockSize = m_setElement(nth + 1, index, dataType, pDst, restSize, pSrc, tshape);
            if (blockSize > 0) {
                memcpy(pDst, pSrc, tailSize);
                restSize -= tailSize;
                pSrc += tailSize;
            }
            pDst += step * tailSize;
        }
    }
    else {
        TP_THROW(VERR_INVALID_CORE);
    }

    return 0;
}

void ETensor::setSlice(VList index, VDataType dataType, int64 datSize, void* value_ptr) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);

    int64 restSize = datSize * TpUtils::byte_size(dataType);

    char* srcPtr = (char*)value_ptr;  // 주소 계산 산술식 사용 위해 형변환 필요
    char* dstPtr = (char*)void_ptr(); // 주소 계산 산술식 사용 위해 형변환 필요

    int64 nBlockSize = m_setElement(0, index, dataType, dstPtr, restSize, srcPtr, m_core->m_shape);

    if (nBlockSize > 0) {
        //memcpy(void_ptr(), value_ptr, byteSize());
        restSize -= byteSize();
    }

    if (restSize != 0) TP_THROW(VERR_UNDEFINED);
}

VValue ETensor::getElement(VList pos) {
    int64 index = 0;

    if (pos.size() > 0) {
        VShape xshape = shape();

        if (pos.size() > xshape.size()) TP_THROW(VERR_SIZE_TENSOR);

        for (int64 n = 0; n < pos.size(); n++) {
            int64 npos = pos[n];
            if (npos < 0 || npos >= xshape[n]) TP_THROW(VERR_OUT_OF_RANGE);
            index = index * xshape[n] + npos;
        }

        for (int64 n = pos.size(); n < xshape.size();  n++) {
            if (xshape[n] != 1) TP_THROW(VERR_UNDEFINED);
        }
    }

    return getElement(index);
}

VValue ETensor::getElement(int64 pos) {
    if (type() == VDataType::int32) {
        int* pElem = int_ptr();
        int nValue = pElem[pos];

        return nValue;
    }
    else if (type() == VDataType::int64) {
        int64* pElem = int64_ptr();
        int64 nValue = pElem[pos];

        return nValue;
    }
    else if (type() == VDataType::float32) {
        float* pElem = float_ptr();
        float fValue = pElem[pos];

        return fValue;
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }
}

int64 m_getElement(int64 nth, VList index, VDataType dataType, char*& pDst, char* pSrc, VShape dstShape, VShape srcShape) {
    if (nth >= index.size()) {
        return (srcShape.size() == 0) ? 1 : srcShape.valid_size();
    }

    VShape tdshape = dstShape.remove_head();
    VShape tsshape = srcShape.remove_head();

    int64 tailSize = TpUtils::byte_size(dataType) * ((tsshape.size() == 0) ? 1 : tsshape.valid_size());

    if (index[nth].is_int64()) {
        pSrc = pSrc + (int64)index[nth] * tailSize;
        return m_getElement(nth + 1, index, dataType, pDst, pSrc, dstShape, tsshape);
    }
    else if (index[nth].is_list()) {
        VList triple = index[nth];

        int64 from = triple[0];
        int64 stop = triple[1];
        int64 step = triple[2];

        if (from < 0) from += srcShape[0];
        if (stop < 0) stop += srcShape[0];

        if (step == 1) {
            char* pStepSrc = pSrc + from * tailSize;
            int64 blockSize = m_getElement(nth + 1, index, dataType, pDst, pStepSrc, tdshape, tsshape);
            if (blockSize > 0) {
                if (from == 0 && stop == srcShape[0] && stop == dstShape[0]) {
                    return blockSize * srcShape[0];
                }
                else {
                    int64 rangeSize = (stop - from) * blockSize * TpUtils::byte_size(dataType);
                    memcpy(pDst, pStepSrc, rangeSize);
                    pDst += rangeSize;
                    return 0;
                }
            }
            from++;    // 시험삼아 첫 원소 시도하는 과정에서 내부 처리 완료, 따라서 첫 원소 스킵
        }

        for (int64 n = from; (step > 0) ? (n < stop) : (n > stop); n += step) {
            char* pStepSrc = pSrc + n * tailSize;
            int64 blockSize = m_getElement(nth + 1, index, dataType, pDst, pStepSrc, tdshape, tsshape);
            if (blockSize > 0) {
                memcpy(pDst, pStepSrc, tailSize);
                pDst += tailSize;
            }
        }
    }
    else {
        TP_THROW(VERR_INVALID_CORE);
    }

    return 0;
}

ETensor ETensor::getSlice(VList index) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);

    VShape dstShape = m_getSliceShape(index, shape());

    ETensor tensor(nn(), dstShape, type());

    char* dstPtr = (char*)tensor.void_ptr();    // 주소 계산 산술식 사용 위해 형변환 필요
    char* srcPtr = (char*)void_ptr();           // 주소 계산 산술식 사용 위해 형변환 필요

    int64 nBlockSize = m_getElement(0, index, type(), dstPtr, srcPtr, dstShape, m_core->m_shape);

    if (nBlockSize > 0) {
        memcpy(tensor.void_ptr(), void_ptr(), byteSize());
    }

    return tensor;
}

VShape ETensor::m_getSliceShape(VList sliceIndex, VShape srcShape) {
    VShape shape;

    for (auto& it : sliceIndex) {
        if (it.is_list()) {
            VList triple = it;
            
            int64 from = triple[0];
            int64 stop = triple[1];
            int64 step = triple[2];

            int64 size = (step > 0) ? ((stop - from - 1) / step + 1) : ((from - stop - 1) / (-step) + 1);

            shape = shape.append(size);
        }
        // else: must be an int index, shape will not be added
    }

    for (int64 n = sliceIndex.size(); n < srcShape.size(); n++) {
        shape = shape.append(srcShape[n]);
    }

    return shape;
}

void ETensor::fetchIdxRows(ETensor src, int64* pnMap, int64 batch_size) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    m_core->m_fetchIdxRows(src.m_core, pnMap, batch_size);
}

void ETensor::fetchIdxRows(ETensor src, int* pnMap, int64 batch_size) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    m_core->m_fetchIdxRows(src.m_core, pnMap, batch_size);
}

void ETensor::copy_into_row(int64 nthRow, ETensor src) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    m_core->m_copy_into_row(nthRow, src);
}

ETensor ETensor::to_type(VDataType type, string option) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    if (m_core->m_type == type) {
        return *this;
    }
    else {
        ETensor tensor(nn(), m_core->m_shape, type);
        tensor.m_core->m_to_type(m_core, option);
        return tensor;
    }
}

ETensor ETensor::set_type(VDataType type, string option) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    return to_type(type, option);
}

void ETensor::save(TpStreamOut* fout) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    
    fout->save_shape(m_core->m_shape);
    fout->save_string(TpUtils::to_string(m_core->m_type));
    fout->save_int(m_core->m_nDevice);

    m_core->m_data.save(fout);
}

ETensor ETensor::load(ENN nn, TpStreamIn* fin) {
    ETensor tensor(nn);
    ETensorCore* core = tensor.m_core;

    core->m_shape = fin->load_shape();
    core->m_type = TpUtils::to_data_type(fin->load_string());
    core->m_nDevice = fin->load_int();

    int64 size = fin->load_int64();

    if (size > 0) {
        core->m_data = ETensorData(nn, size);
        core->m_data.readFrom(fin);
    }

    return tensor;
}

ETensor ETensor::load(ENN nn, FILE* fin) {
    ETensor tensor(nn);
    ETensorCore* core = tensor.m_core;

    core->m_shape = TpStreamIn::load_shape(fin);
    core->m_type = TpUtils::to_data_type(TpStreamIn::load_string(fin));
    core->m_nDevice = TpStreamIn::load_int(fin);

    int64 size = TpStreamIn::load_int64(fin);

    core->m_data = ETensorData(nn, size);
    core->m_data.readFrom(fin);

    return tensor;
}

ETensor ETensor::reshape(VShape shape) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    int64 fill_size = -1;
    
    if (shape.valid_size() < 0) {
        fill_size = this->shape().valid_size() / -shape.valid_size();
    }

    for (int64 n = 0; n < shape.size(); n++) {
        if (shape[n] == -1 && fill_size > 0) {
            shape[n] = fill_size;
            fill_size = -1;
        }
        else if (shape[n] <= 0) {
            TP_THROW2(VERR_SHAPE_TENSOR, "reshape");
        }
    }

    if (shape.valid_size() != this->shape().valid_size()) {
        TP_THROW(VERR_SIZE_TENSOR);
    }

    ETensor tensor(*this, shape);
    return tensor;
}

ETensor ETensor::ones(ENN nn, VShape shape) {
    ETensor y(nn, shape, VDataType::float32);

    float* py = y.float_ptr();
    int64 ndat = shape.valid_size();

    for (int64 n = 0; n < ndat; n++) {
        py[n] = 1;
    }

    return y;
}

ETensor ETensor::zeros(ENN nn, VShape shape) {
    ETensor y(nn, shape, VDataType::float32);

    float* py = y.float_ptr();
    int64 ndat = shape.valid_size();

    for (int64 n = 0; n < ndat; n++) {
        py[n] = 0;
    }

    return y;
}

ETensor ETensor::rand_uniform(ENN nn, VShape shape, float min, float max) {
    ETensor y(nn, shape, VDataType::float32);

    std::uniform_real_distribution<float> coin(min, max);

    float* py = y.float_ptr();
    int64 ndat = shape.valid_size();

    for (int64 n = 0; n < ndat; n++) {
        py[n] = coin(ms_randGen);
    }

    return y;
}

void ETensor::rand_uniform_on(float min, float max) {
    ETensor y = *this;

    std::uniform_real_distribution<float> coin(min, max);

    float* py = y.float_ptr();
    int64 ndat = y.shape().valid_size();

    for (int64 n = 0; n < ndat; n++) {
        py[n] = coin(ms_randGen);
    }
}

ETensor ETensor::rand_normal(ENN nn, VShape shape, float avg, float std) {
    ETensor y(nn, shape, VDataType::float32);

    std::normal_distribution<float> coin(avg, std);

    float* py = y.float_ptr();
    int64 ndat = shape.valid_size();

    for (int64 n = 0; n < ndat; n++) {
        py[n] = coin(ms_randGen);
    }

    return y;
}

ETensor ETensor::rand_onehot(ENN nn, VShape shape) {
    ETensor y = ETensor::zeros(nn, shape);

    float* py = y.float_ptr();

    int64 ndat = shape.valid_size();
    int64 ncol = shape[-1];

    std::uniform_int_distribution<int> coin(0, (int)ncol-1);

    for (int64 n = 0; n < ndat; n += ncol) {
        int dice = coin(ms_randGen);
        py[n+dice] = 1.0f;
    }

    return y;
}

ETensor ETensor::arange(ENN nn, VValue arg1, VValue arg2, VValue arg3, VDict kwArgs) {
    VDataType type = VDataType::int32;

    int64 istart, istop, istep;
    float fstart, fstop, fstep;

    if (arg1.is_none()) {
        TP_THROW(VERR_INVALID_ARGUMENT);
    }
    else if (arg1.is_int32()) {
        istart = arg1;
        fstart = arg1;
    }
    else if (arg1.is_int64()) {
        istart = arg1;
        fstart = arg1;
        type = VDataType::int64;
    }
    else if (arg1.is_float()) {
        istart = arg1;
        fstart = arg1;
        type = VDataType::float32;
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }

    if (arg2.is_none()) {
        istop = istart;
        fstop = fstart;
        istart = 0;
        fstart = 0;
    }
    else if (arg2.is_int32()) {
        istop = arg2;
        fstop = arg2;
    }
    else if (arg2.is_int64()) {
        istop = arg2;
        fstop = arg2;
        if (type != VDataType::float32) type = VDataType::int64;
    }
    else if (arg2.is_float()) {
        istop = arg2;
        fstop = arg2;
        type = VDataType::float32;
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }

    if (arg3.is_none()) {
        istep = 1;
        fstep = 1;
    }
    else if (arg3.is_int32()) {
        istep = arg3;
        fstep = arg3;
    }
    else if (arg3.is_int64()) {
        istep = arg3;
        fstep = arg3;
        if (type != VDataType::float32) type = VDataType::int64;
    }
    else if (arg3.is_float()) {
        istep = arg3;
        fstep = arg3;
        type = VDataType::float32;
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }

    if (kwArgs.find("dtype") != kwArgs.end()) {
        type = TpUtils::to_data_type(kwArgs["type"]);
        if (type != VDataType::int32 && type != VDataType::int64 && type != VDataType::float32) {
            TP_THROW(VERR_UNDEFINED);
        }
    }

    if (type == VDataType::float32) {
        if (fstart == fstop || fstep == 0) TP_THROW(VERR_UNDEFINED);

        if (fstart < fstop && fstep < 0) TP_THROW(VERR_UNDEFINED);
        if (fstart > fstop && fstep > 0) TP_THROW(VERR_UNDEFINED);

        int64 cnt = (int64)((fstop - fstart) / fstep);
        if (fstop != fstart + cnt * fstep) cnt++;

        ETensor y(nn, VShape{ cnt }, type);

        float* py = y.float_ptr();

        for (int64 n = 0; n < cnt; n++) {
            py[n] = fstart;
            fstart += fstep;
        }

        return y;
    }
    else if (type == VDataType::int64) {
        if (istart == istop || istep == 0) TP_THROW(VERR_UNDEFINED);

        if (istart < istop && istep < 0) TP_THROW(VERR_UNDEFINED);
        if (istart > istop && istep > 0) TP_THROW(VERR_UNDEFINED);

        int64 cnt = (int64)((istop - istart) / istep);
        if (istop != istart + cnt * istep) cnt++;

        ETensor y(nn, VShape{ cnt }, type);

        int64* py = y.int64_ptr();

        for (int64 n = 0; n < cnt; n++) {
            py[n] = istart;
            istart += istep;
        }

        return y;
    }
    else if (type == VDataType::int32) {
        if (istart == istop || istep == 0) TP_THROW(VERR_UNDEFINED);

        if (istart < istop && istep < 0) TP_THROW(VERR_UNDEFINED);
        if (istart > istop && istep > 0) TP_THROW(VERR_UNDEFINED);

        int64 cnt = (int64)((istop - istart) / istep);
        if (istop != istart + cnt * istep) cnt++;

        ETensor y(nn, VShape{ cnt }, type);

        int* py = y.int_ptr();

        for (int64 n = 0; n < cnt; n++) {
            py[n] = (int)istart;
            istart += istep;
        }

        return y;
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }
}

ETensor ETensor::matmul(ETensor x1, ETensor x2) {
    if (!x1.isValid() || !x2.isValid()) TP_THROW(VERR_INVALID_CORE);
    if (x1.type() != VDataType::float32 || x2.type() != VDataType::float32) TP_THROW(VERR_NOT_IMPLEMENTED_YET);

    VShape shape1 = x1.shape();
    VShape shape2 = x2.shape();

    if (shape1.size() != 2 || shape2.size() != 2) TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    if (shape1[1] != shape2[0]) TP_THROW2(VERR_SHAPE_TENSOR, "matmul");

    int64 nrow = shape1[0];
    int64 nvec = shape1[1];
    int64 ncol = shape2[1];

    ETensor y(x1.nn(), VShape{ nrow, ncol }, x1.type());

    for (int64 nr = 0; nr < nrow; nr++) {
        for (int64 nc = 0; nc < ncol; nc++) {
            float sum = 0;
            for (int64 nv = 0; nv < nvec; nv++) {
                int64 xpos1 = nr * nvec + nv;
                int64 xpos2 = nv * ncol + nc;

                sum += (float)x1.getElement(xpos1) * (float)x2.getElement(xpos2);
            }
            int64 ypos = nr * ncol + nc;
            y.setElement(ypos, sum);
        }
    }

    return y;
}

ETensor ETensor::hconcat(ETensor x1, ETensor x2) {
    if (!x1.isValid() || !x2.isValid()) TP_THROW(VERR_INVALID_CORE);
    if (x1.type() != VDataType::float32 || x2.type() != VDataType::float32) TP_THROW(VERR_NOT_IMPLEMENTED_YET);

    VShape shape1 = x1.shape();
    VShape shape2 = x2.shape();

    if (shape1.remove_end() != shape2.remove_end()) TP_THROW2(VERR_SHAPE_TENSOR, "hconcat");

    int64 nvec1 = shape1[-1];
    int64 nvec2 = shape2[-1];

    int64 nrow = shape1.valid_size() / nvec1;

    VShape yshape = shape1.replace_end(nvec1 + nvec2);

    ETensor y(x1.nn(), yshape, x1.type());

    for (int64 nr = 0; nr < nrow; nr++) {
        for (int64 nv = 0; nv < nvec1; nv++) {
            int64 xpos1 = nr * nvec1 + nv;
            int64 ypos = nr * (nvec1 + nvec2) + nv;
            y.setElement(ypos, x1.getElement(xpos1));
        }
        for (int64 nv = 0; nv < nvec2; nv++) {
            int64 xpos2 = nr * nvec2 + nv;
            int64 ypos = nr * (nvec1 + nvec2) + (nvec1 + nv);
            y.setElement(ypos, x2.getElement(xpos2));
        }
    }

    return y;
}

ETensor ETensor::vstack(ETensor x1, ETensor x2) {
    VShape shape1 = x1.shape();
    VShape shape2 = x2.shape();

    int64 nbat1 = shape1[0];
    int64 nbat2 = shape2[0];

    VShape yshape = shape1.replace_head(nbat1 + nbat2);

    ETensor y(x1.nn(), yshape, x1.type());

    y.vstack_on(x1, x2);

    return y;
}

void ETensor::vstack_on(ETensor x1, ETensor x2) {
    if (!isValid() || !x1.isValid() || !x2.isValid()) TP_THROW(VERR_INVALID_CORE);
    if (type() != VDataType::float32 || x1.type() != VDataType::float32 || x2.type() != VDataType::float32) TP_THROW(VERR_NOT_IMPLEMENTED_YET);

    VShape yshape = shape();
    VShape shape1 = x1.shape();
    VShape shape2 = x2.shape();

    if (shape1.remove_head() != shape2.remove_head()) TP_THROW2(VERR_SHAPE_TENSOR, "vstack_on");
    if (shape1.remove_head() != yshape.remove_head()) TP_THROW2(VERR_SHAPE_TENSOR, "vstack_on");

    int64 nbat = yshape[0];
    int64 nbat1 = shape1[0];
    int64 nbat2 = shape2[0];

    int64 ndat = shape1.valid_size() / nbat1;

    if (nbat != nbat1 + nbat2) TP_THROW2(VERR_SHAPE_TENSOR, "vstack_on");

    float* py = float_ptr();
    float* px1 = x1.float_ptr();
    float* px2 = x2.float_ptr();

    memcpy(py, px1, sizeof(float) * shape1.valid_size());

    py += shape1.valid_size();

    memcpy(py, px2, sizeof(float) * shape2.valid_size());
}

ETensor ETensor::tile(ETensor x, int64 rep) {
    if (!x.isValid()) TP_THROW(VERR_INVALID_CORE);

    int64 nsize = x.shape().valid_size();

    ETensor y(x.nn(), { nsize * rep }, x.type());

    for (int64 nx = 0; nx < nsize; nx++) {
        VValue val = x.getElement(nx);
        for (int64 nr = 0; nr < rep; nr++) {
            int64 ypos = nr * nsize + nx;
            y.setElement(ypos, val);
        }
    }

    return y;
}

ETensor ETensor::repeat(ETensor x, int64 rep) {
    if (!x.isValid()) TP_THROW(VERR_INVALID_CORE);

    int64 nsize = x.shape().valid_size();

    ETensor y(x.nn(), { nsize * rep }, x.type());

    for (int64 nx = 0; nx < nsize; nx++) {
        VValue val = x.getElement(nx);
        for (int64 nr = 0; nr < rep; nr++) {
            int64 ypos = nx * rep + nr;
            y.setElement(ypos, val);
        }
    }

    return y;
}

void ETensor::shuffle() {
    ETensor x = *this;

    int64 size = x.shape().valid_size();
    VDataType type = x.type();

    if (type == VDataType::float32) {
        float* px = x.float_ptr();

        for (int64 n = 0; n < size; n++) {
            int64 idx = rand() % size;
            float tmp = px[n];
            px[n] = px[idx];
            px[idx] = tmp;
        }
    }
    else if (type == VDataType::int32) {
        int* px = x.int_ptr();

        for (int64 n = 0; n < size; n++) {
            int64 idx = rand() % size;
            int tmp = px[n];
            px[n] = px[idx];
            px[idx] = tmp;
        }
    }
    else if (type == VDataType::int64) {
        int64* px = x.int64_ptr();

        for (int64 n = 0; n < size; n++) {
            int64 idx = rand() % size;
            int64 tmp = px[n];
            px[n] = px[idx];
            px[idx] = tmp;
        }
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }
}

ETensor ETensor::operator ==(ETensor rhs) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    // 예제에 나타난 산술연산은 서버 엔진 기능을 호출하지 않고 바로 처리한다.
    // 이는 예제에 표현되어 있음은 numpy 등을 이용해 게산한다는 의미를 갖기 때문이먀
    // 클라이언트 단의 연산이 가능하도록 데이터 다운로드 상태를 확인해야 하기 때문이기도 하다.

    VShape lshape = shape();
    VShape rshape = rhs.shape();
    
    VDataType rtype = rhs.type();

    while (lshape.size() > 1 && lshape[-1] == 1) lshape = lshape.remove_end();
    while (rshape.size() > 1 && rshape[-1] == 1) rshape = rshape.remove_end();

    if (lshape != rshape) TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    if (type() != rtype) TP_THROW(VERR_NOT_IMPLEMENTED_YET);

    ETensor y(nn(), rshape, VDataType::bool8);   // 32bit 정수형, 64비트 정수형 numpy 참고해 추후 결정

    int64 ndat = rshape.valid_size();

    unsigned char* pDst = y.bool_ptr();

    if (rtype == VDataType::int32) {
        int* pl = int_ptr();
        int* pr = rhs.int_ptr();

        for (int64 n = 0; n < ndat; n++) {
            pDst[n] = (pl[n] == pr[n]) ? 1 : 0;
        }
    }
    else if (rtype == VDataType::int64) {
        int64* pl = int64_ptr();
        int64* pr = rhs.int64_ptr();

        for (int64 n = 0; n < ndat; n++) {
            pDst[n] = (pl[n] == pr[n]) ? 1 : 0;
        }
    }
    else if (rtype == VDataType::bool8) {
        unsigned char* pl = bool_ptr();
        unsigned char* pr = rhs.bool_ptr();

        for (int64 n = 0; n < ndat; n++) {
            pDst[n] = (pl[n] == pr[n]) ? 1 : 0;
        }
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }

    return y;
}

ETensor ETensor::operator ==(int value) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    VShape xshape = shape();
    VDataType xtype = type();

    ETensor y(nn(), xshape, VDataType::bool8);   // 32bit 정수형, 64비트 정수형 numpy 참고해 추후 결정

    int64 ndat = xshape.valid_size();

    unsigned char* pDst = y.bool_ptr();

    if (xtype == VDataType::int32) {
        int* pl = int_ptr();

        for (int64 n = 0; n < ndat; n++) {
            pDst[n] = (pl[n] == value) ? 1 : 0;
        }
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }

    return y;
}

ETensor ETensor::operator ==(int64 value) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    VShape xshape = shape();
    VDataType xtype = type();

    ETensor y(nn(), xshape, VDataType::bool8);   // 32bit 정수형, 64비트 정수형 numpy 참고해 추후 결정

    int64 ndat = xshape.valid_size();

    unsigned char* pDst = y.bool_ptr();

    if (xtype == VDataType::int64) {
        int64* pl = int64_ptr();

        for (int64 n = 0; n < ndat; n++) {
            pDst[n] = (pl[n] == value) ? 1 : 0;
        }
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }

    return y;
}

ETensor ETensor::operator ==(float value) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    VShape xshape = shape();
    VDataType xtype = type();

    ETensor y(nn(), xshape, VDataType::bool8);   // 32bit 정수형, 64비트 정수형 numpy 참고해 추후 결정

    int64 ndat = xshape.valid_size();

    unsigned char* pDst = y.bool_ptr();

    if (xtype == VDataType::float32) {
        float* pl = float_ptr();

        for (int64 n = 0; n < ndat; n++) {
            pDst[n] = (pl[n] == value) ? 1 : 0;
        }
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }

    return y;
}

ETensor ETensor::operator !=(ETensor rhs) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    // 예제에 나타난 산술연산은 서버 엔진 기능을 호출하지 않고 바로 처리한다.
    // 이는 예제에 표현되어 있음은 numpy 등을 이용해 게산한다는 의미를 갖기 때문이먀
    // 클라이언트 단의 연산이 가능하도록 데이터 다운로드 상태를 확인해야 하기 때문이기도 하다.

    VShape lshape = shape();
    VShape rshape = rhs.shape();

    VDataType rtype = rhs.type();

    while (lshape.size() > 1 && lshape[-1] == 1) lshape = lshape.remove_end();
    while (rshape.size() > 1 && rshape[-1] == 1) rshape = rshape.remove_end();

    if (lshape != rshape) TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    if (type() != rtype) TP_THROW(VERR_NOT_IMPLEMENTED_YET);

    ETensor y(nn(), rshape, VDataType::bool8);   // 32bit 정수형, 64비트 정수형 numpy 참고해 추후 결정

    int64 ndat = rshape.valid_size();

    unsigned char* pDst = y.bool_ptr();

    if (rtype == VDataType::int32) {
        int* pl = int_ptr();
        int* pr = rhs.int_ptr();

        for (int64 n = 0; n < ndat; n++) {
            pDst[n] = (pl[n] != pr[n]) ? 1 : 0;
        }
    }
    else if (rtype == VDataType::int64) {
        int64* pl = int64_ptr();
        int64* pr = rhs.int64_ptr();

        for (int64 n = 0; n < ndat; n++) {
            pDst[n] = (pl[n] != pr[n]) ? 1 : 0;
        }
    }
    else if (rtype == VDataType::bool8) {
        unsigned char* pl = bool_ptr();
        unsigned char* pr = rhs.bool_ptr();

        for (int64 n = 0; n < ndat; n++) {
            pDst[n] = (pl[n] != pr[n]) ? 1 : 0;
        }
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }

    return y;
}

ETensor ETensor::operator !=(int value) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    VShape xshape = shape();
    VDataType xtype = type();

    ETensor y(nn(), xshape, VDataType::bool8);   // 32bit 정수형, 64비트 정수형 numpy 참고해 추후 결정

    int64 ndat = xshape.valid_size();

    unsigned char* pDst = y.bool_ptr();

    if (xtype == VDataType::int32) {
        int* pl = int_ptr();

        for (int64 n = 0; n < ndat; n++) {
            pDst[n] = (pl[n] != value) ? 1 : 0;
        }
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }

    return y;
}

ETensor ETensor::operator !=(int64 value) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    VShape xshape = shape();
    VDataType xtype = type();

    ETensor y(nn(), xshape, VDataType::bool8);   // 32bit 정수형, 64비트 정수형 numpy 참고해 추후 결정

    int64 ndat = xshape.valid_size();

    unsigned char* pDst = y.bool_ptr();

    if (xtype == VDataType::int64) {
        int64* pl = int64_ptr();

        for (int64 n = 0; n < ndat; n++) {
            pDst[n] = (pl[n] != value) ? 1 : 0;
        }
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }

    return y;
}

ETensor ETensor::operator !=(float value) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    VShape xshape = shape();
    VDataType xtype = type();

    ETensor y(nn(), xshape, VDataType::bool8);   // 32bit 정수형, 64비트 정수형 numpy 참고해 추후 결정

    int64 ndat = xshape.valid_size();

    unsigned char* pDst = y.bool_ptr();

    if (xtype == VDataType::float32) {
        float* pl = float_ptr();

        for (int64 n = 0; n < ndat; n++) {
            pDst[n] = (pl[n] != value) ? 1 : 0;
        }
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }

    return y;
}

ETensor ETensor::operator >(ETensor rhs) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    // 예제에 나타난 산술연산은 서버 엔진 기능을 호출하지 않고 바로 처리한다.
    // 이는 예제에 표현되어 있음은 numpy 등을 이용해 게산한다는 의미를 갖기 때문이먀
    // 클라이언트 단의 연산이 가능하도록 데이터 다운로드 상태를 확인해야 하기 때문이기도 하다.

    VShape lshape = shape();
    VShape rshape = rhs.shape();

    VDataType rtype = rhs.type();

    while (lshape.size() > 1 && lshape[-1] == 1) lshape = lshape.remove_end();
    while (rshape.size() > 1 && rshape[-1] == 1) rshape = rshape.remove_end();

    if (lshape != rshape) TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    if (type() != rtype) TP_THROW(VERR_NOT_IMPLEMENTED_YET);

    ETensor y(nn(), rshape, VDataType::bool8);   // 32bit 정수형, 64비트 정수형 numpy 참고해 추후 결정

    int64 ndat = rshape.valid_size();

    unsigned char* pDst = y.bool_ptr();

    if (rtype == VDataType::int32) {
        int* pl = int_ptr();
        int* pr = rhs.int_ptr();

        for (int64 n = 0; n < ndat; n++) {
            pDst[n] = (pl[n] > pr[n]) ? 1 : 0;
        }
    }
    else if (rtype == VDataType::int64) {
        int64* pl = int64_ptr();
        int64* pr = rhs.int64_ptr();

        for (int64 n = 0; n < ndat; n++) {
            pDst[n] = (pl[n] > pr[n]) ? 1 : 0;
        }
    }
    else if (rtype == VDataType::bool8) {
        unsigned char* pl = bool_ptr();
        unsigned char* pr = rhs.bool_ptr();

        for (int64 n = 0; n < ndat; n++) {
            pDst[n] = (pl[n] > pr[n]) ? 1 : 0;
        }
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }

    return y;
}

ETensor ETensor::operator >(int value) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    VShape xshape = shape();
    VDataType xtype = type();

    ETensor y(nn(), xshape, VDataType::bool8);   // 32bit 정수형, 64비트 정수형 numpy 참고해 추후 결정

    int64 ndat = xshape.valid_size();

    unsigned char* pDst = y.bool_ptr();

    if (xtype == VDataType::int32) {
        int* pl = int_ptr();

        for (int64 n = 0; n < ndat; n++) {
            pDst[n] = (pl[n] > value) ? 1 : 0;
        }
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }

    return y;
}

ETensor ETensor::operator >(int64 value) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    VShape xshape = shape();
    VDataType xtype = type();

    ETensor y(nn(), xshape, VDataType::bool8);   // 32bit 정수형, 64비트 정수형 numpy 참고해 추후 결정

    int64 ndat = xshape.valid_size();

    unsigned char* pDst = y.bool_ptr();

    if (xtype == VDataType::int64) {
        int64* pl = int64_ptr();

        for (int64 n = 0; n < ndat; n++) {
            pDst[n] = (pl[n] > value) ? 1 : 0;
        }
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }

    return y;
}

ETensor ETensor::operator >(float value) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    VShape xshape = shape();
    VDataType xtype = type();

    ETensor y(nn(), xshape, VDataType::bool8);   // 32bit 정수형, 64비트 정수형 numpy 참고해 추후 결정

    int64 ndat = xshape.valid_size();

    unsigned char* pDst = y.bool_ptr();

    if (xtype == VDataType::float32) {
        float* pl = float_ptr();

        for (int64 n = 0; n < ndat; n++) {
            pDst[n] = (pl[n] > value) ? 1 : 0;
        }
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }

    return y;
}

ETensor ETensor::operator >=(ETensor rhs) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    // 예제에 나타난 산술연산은 서버 엔진 기능을 호출하지 않고 바로 처리한다.
    // 이는 예제에 표현되어 있음은 numpy 등을 이용해 게산한다는 의미를 갖기 때문이먀
    // 클라이언트 단의 연산이 가능하도록 데이터 다운로드 상태를 확인해야 하기 때문이기도 하다.

    VShape lshape = shape();
    VShape rshape = rhs.shape();

    VDataType rtype = rhs.type();

    while (lshape.size() > 1 && lshape[-1] == 1) lshape = lshape.remove_end();
    while (rshape.size() > 1 && rshape[-1] == 1) rshape = rshape.remove_end();

    if (lshape != rshape) TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    if (type() != rtype) TP_THROW(VERR_NOT_IMPLEMENTED_YET);

    ETensor y(nn(), rshape, VDataType::bool8);   // 32bit 정수형, 64비트 정수형 numpy 참고해 추후 결정

    int64 ndat = rshape.valid_size();

    unsigned char* pDst = y.bool_ptr();

    if (rtype == VDataType::int32) {
        int* pl = int_ptr();
        int* pr = rhs.int_ptr();

        for (int64 n = 0; n < ndat; n++) {
            pDst[n] = (pl[n] >= pr[n]) ? 1 : 0;
        }
    }
    else if (rtype == VDataType::int64) {
        int64* pl = int64_ptr();
        int64* pr = rhs.int64_ptr();

        for (int64 n = 0; n < ndat; n++) {
            pDst[n] = (pl[n] >= pr[n]) ? 1 : 0;
        }
    }
    else if (rtype == VDataType::bool8) {
        unsigned char* pl = bool_ptr();
        unsigned char* pr = rhs.bool_ptr();

        for (int64 n = 0; n < ndat; n++) {
            pDst[n] = (pl[n] >= pr[n]) ? 1 : 0;
        }
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }

    return y;
}

ETensor ETensor::operator >=(int value) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    VShape xshape = shape();
    VDataType xtype = type();

    ETensor y(nn(), xshape, VDataType::bool8);   // 32bit 정수형, 64비트 정수형 numpy 참고해 추후 결정

    int64 ndat = xshape.valid_size();

    unsigned char* pDst = y.bool_ptr();

    if (xtype == VDataType::int32) {
        int* pl = int_ptr();

        for (int64 n = 0; n < ndat; n++) {
            pDst[n] = (pl[n] >= value) ? 1 : 0;
        }
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }

    return y;
}

ETensor ETensor::operator >=(int64 value) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    VShape xshape = shape();
    VDataType xtype = type();

    ETensor y(nn(), xshape, VDataType::bool8);   // 32bit 정수형, 64비트 정수형 numpy 참고해 추후 결정

    int64 ndat = xshape.valid_size();

    unsigned char* pDst = y.bool_ptr();

    if (xtype == VDataType::int64) {
        int64* pl = int64_ptr();

        for (int64 n = 0; n < ndat; n++) {
            pDst[n] = (pl[n] >= value) ? 1 : 0;
        }
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }

    return y;
}

ETensor ETensor::operator >=(float value) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    VShape xshape = shape();
    VDataType xtype = type();

    ETensor y(nn(), xshape, VDataType::bool8);   // 32bit 정수형, 64비트 정수형 numpy 참고해 추후 결정

    int64 ndat = xshape.valid_size();

    unsigned char* pDst = y.bool_ptr();

    if (xtype == VDataType::float32) {
        float* pl = float_ptr();

        for (int64 n = 0; n < ndat; n++) {
            pDst[n] = (pl[n] >= value) ? 1 : 0;
        }
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }

    return y;
}

ETensor ETensor::operator <(ETensor rhs) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    // 예제에 나타난 산술연산은 서버 엔진 기능을 호출하지 않고 바로 처리한다.
    // 이는 예제에 표현되어 있음은 numpy 등을 이용해 게산한다는 의미를 갖기 때문이먀
    // 클라이언트 단의 연산이 가능하도록 데이터 다운로드 상태를 확인해야 하기 때문이기도 하다.

    VShape lshape = shape();
    VShape rshape = rhs.shape();

    VDataType rtype = rhs.type();

    while (lshape.size() > 1 && lshape[-1] == 1) lshape = lshape.remove_end();
    while (rshape.size() > 1 && rshape[-1] == 1) rshape = rshape.remove_end();

    if (lshape != rshape) TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    if (type() != rtype) TP_THROW(VERR_NOT_IMPLEMENTED_YET);

    ETensor y(nn(), rshape, VDataType::bool8);   // 32bit 정수형, 64비트 정수형 numpy 참고해 추후 결정

    int64 ndat = rshape.valid_size();

    unsigned char* pDst = y.bool_ptr();

    if (rtype == VDataType::int32) {
        int* pl = int_ptr();
        int* pr = rhs.int_ptr();

        for (int64 n = 0; n < ndat; n++) {
            pDst[n] = (pl[n] < pr[n]) ? 1 : 0;
        }
    }
    else if (rtype == VDataType::int64) {
        int64* pl = int64_ptr();
        int64* pr = rhs.int64_ptr();

        for (int64 n = 0; n < ndat; n++) {
            pDst[n] = (pl[n] < pr[n]) ? 1 : 0;
        }
    }
    else if (rtype == VDataType::bool8) {
        unsigned char* pl = bool_ptr();
        unsigned char* pr = rhs.bool_ptr();

        for (int64 n = 0; n < ndat; n++) {
            pDst[n] = (pl[n] < pr[n]) ? 1 : 0;
        }
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }

    return y;
}

ETensor ETensor::operator <(int value) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    VShape xshape = shape();
    VDataType xtype = type();

    ETensor y(nn(), xshape, VDataType::bool8);   // 32bit 정수형, 64비트 정수형 numpy 참고해 추후 결정

    int64 ndat = xshape.valid_size();

    unsigned char* pDst = y.bool_ptr();

    if (xtype == VDataType::int32) {
        int* pl = int_ptr();

        for (int64 n = 0; n < ndat; n++) {
            pDst[n] = (pl[n] < value) ? 1 : 0;
        }
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }

    return y;
}

ETensor ETensor::operator <(int64 value) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    VShape xshape = shape();
    VDataType xtype = type();

    ETensor y(nn(), xshape, VDataType::bool8);   // 32bit 정수형, 64비트 정수형 numpy 참고해 추후 결정

    int64 ndat = xshape.valid_size();

    unsigned char* pDst = y.bool_ptr();

    if (xtype == VDataType::int64) {
        int64* pl = int64_ptr();

        for (int64 n = 0; n < ndat; n++) {
            pDst[n] = (pl[n] < value) ? 1 : 0;
        }
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }

    return y;
}

ETensor ETensor::operator <(float value) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    VShape xshape = shape();
    VDataType xtype = type();

    ETensor y(nn(), xshape, VDataType::bool8);   // 32bit 정수형, 64비트 정수형 numpy 참고해 추후 결정

    int64 ndat = xshape.valid_size();

    unsigned char* pDst = y.bool_ptr();

    if (xtype == VDataType::float32) {
        float* pl = float_ptr();

        for (int64 n = 0; n < ndat; n++) {
            pDst[n] = (pl[n] < value) ? 1 : 0;
        }
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }

    return y;
}

ETensor ETensor::operator <=(ETensor rhs) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    // 예제에 나타난 산술연산은 서버 엔진 기능을 호출하지 않고 바로 처리한다.
    // 이는 예제에 표현되어 있음은 numpy 등을 이용해 게산한다는 의미를 갖기 때문이먀
    // 클라이언트 단의 연산이 가능하도록 데이터 다운로드 상태를 확인해야 하기 때문이기도 하다.

    VShape lshape = shape();
    VShape rshape = rhs.shape();

    VDataType rtype = rhs.type();

    while (lshape.size() > 1 && lshape[-1] == 1) lshape = lshape.remove_end();
    while (rshape.size() > 1 && rshape[-1] == 1) rshape = rshape.remove_end();

    if (lshape != rshape) TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    if (type() != rtype) TP_THROW(VERR_NOT_IMPLEMENTED_YET);

    ETensor y(nn(), rshape, VDataType::bool8);   // 32bit 정수형, 64비트 정수형 numpy 참고해 추후 결정

    int64 ndat = rshape.valid_size();

    unsigned char* pDst = y.bool_ptr();

    if (rtype == VDataType::int32) {
        int* pl = int_ptr();
        int* pr = rhs.int_ptr();

        for (int64 n = 0; n < ndat; n++) {
            pDst[n] = (pl[n] <= pr[n]) ? 1 : 0;
        }
    }
    else if (rtype == VDataType::int64) {
        int64* pl = int64_ptr();
        int64* pr = rhs.int64_ptr();

        for (int64 n = 0; n < ndat; n++) {
            pDst[n] = (pl[n] <= pr[n]) ? 1 : 0;
        }
    }
    else if (rtype == VDataType::bool8) {
        unsigned char* pl = bool_ptr();
        unsigned char* pr = rhs.bool_ptr();

        for (int64 n = 0; n < ndat; n++) {
            pDst[n] = (pl[n] <= pr[n]) ? 1 : 0;
        }
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }

    return y;
}

ETensor ETensor::operator <=(int value) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    VShape xshape = shape();
    VDataType xtype = type();

    ETensor y(nn(), xshape, VDataType::bool8);   // 32bit 정수형, 64비트 정수형 numpy 참고해 추후 결정

    int64 ndat = xshape.valid_size();

    unsigned char* pDst = y.bool_ptr();

    if (xtype == VDataType::int32) {
        int* pl = int_ptr();

        for (int64 n = 0; n < ndat; n++) {
            pDst[n] = (pl[n] <= value) ? 1 : 0;
        }
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }

    return y;
}

ETensor ETensor::operator <=(int64 value) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    VShape xshape = shape();
    VDataType xtype = type();

    ETensor y(nn(), xshape, VDataType::bool8);   // 32bit 정수형, 64비트 정수형 numpy 참고해 추후 결정

    int64 ndat = xshape.valid_size();

    unsigned char* pDst = y.bool_ptr();

    if (xtype == VDataType::int64) {
        int64* pl = int64_ptr();

        for (int64 n = 0; n < ndat; n++) {
            pDst[n] = (pl[n] <= value) ? 1 : 0;
        }
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }

    return y;
}

ETensor ETensor::operator <=(float value) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    VShape xshape = shape();
    VDataType xtype = type();

    ETensor y(nn(), xshape, VDataType::bool8);   // 32bit 정수형, 64비트 정수형 numpy 참고해 추후 결정

    int64 ndat = xshape.valid_size();

    unsigned char* pDst = y.bool_ptr();

    if (xtype == VDataType::float32) {
        float* pl = float_ptr();

        for (int64 n = 0; n < ndat; n++) {
            pDst[n] = (pl[n] <= value) ? 1 : 0;
        }
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }

    return y;
}

ETensor ETensor::operator &&(ETensor rhs) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    // 예제에 나타난 산술연산은 서버 엔진 기능을 호출하지 않고 바로 처리한다.
    // 이는 예제에 표현되어 있음은 numpy 등을 이용해 게산한다는 의미를 갖기 때문이먀
    // 클라이언트 단의 연산이 가능하도록 데이터 다운로드 상태를 확인해야 하기 때문이기도 하다.

    VShape lshape = shape();
    VShape rshape = rhs.shape();

    VDataType rtype = rhs.type();

    while (lshape.size() > 1 && lshape[-1] == 1) lshape = lshape.remove_end();
    while (rshape.size() > 1 && rshape[-1] == 1) rshape = rshape.remove_end();

    if (lshape != rshape) TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    if (type() != rtype) TP_THROW(VERR_NOT_IMPLEMENTED_YET);

    ETensor y(nn(), rshape, VDataType::bool8);   // 32bit 정수형, 64비트 정수형 numpy 참고해 추후 결정

    int64 ndat = rshape.valid_size();

    unsigned char* pDst = y.bool_ptr();

    if (rtype == VDataType::bool8) {
        unsigned char* pl = bool_ptr();
        unsigned char* pr = rhs.bool_ptr();

        for (int64 n = 0; n < ndat; n++) {
            pDst[n] = (pl[n] != 0) && (pr[n] != 0);
        }
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }

    return y;
}

ETensor ETensor::operator ||(ETensor rhs) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    // 예제에 나타난 산술연산은 서버 엔진 기능을 호출하지 않고 바로 처리한다.
    // 이는 예제에 표현되어 있음은 numpy 등을 이용해 게산한다는 의미를 갖기 때문이먀
    // 클라이언트 단의 연산이 가능하도록 데이터 다운로드 상태를 확인해야 하기 때문이기도 하다.

    VShape lshape = shape();
    VShape rshape = rhs.shape();

    VDataType rtype = rhs.type();

    while (lshape.size() > 1 && lshape[-1] == 1) lshape = lshape.remove_end();
    while (rshape.size() > 1 && rshape[-1] == 1) rshape = rshape.remove_end();

    if (lshape != rshape) TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    if (type() != rtype) TP_THROW(VERR_NOT_IMPLEMENTED_YET);

    ETensor y(nn(), rshape, VDataType::bool8);   // 32bit 정수형, 64비트 정수형 numpy 참고해 추후 결정

    int64 ndat = rshape.valid_size();

    unsigned char* pDst = y.bool_ptr();

    if (rtype == VDataType::bool8) {
        unsigned char* pl = bool_ptr();
        unsigned char* pr = rhs.bool_ptr();

        for (int64 n = 0; n < ndat; n++) {
            pDst[n] = (pl[n] != 0) || (pr[n] != 0);
        }
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }

    return y;
}

ETensor ETensor::operator &(ETensor rhs) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    // 예제에 나타난 산술연산은 서버 엔진 기능을 호출하지 않고 바로 처리한다.
    // 이는 예제에 표현되어 있음은 numpy 등을 이용해 게산한다는 의미를 갖기 때문이먀
    // 클라이언트 단의 연산이 가능하도록 데이터 다운로드 상태를 확인해야 하기 때문이기도 하다.

    VShape lshape = shape();
    VShape rshape = rhs.shape();

    VDataType rtype = rhs.type();

    while (lshape.size() > 1 && lshape[-1] == 1) lshape = lshape.remove_end();
    while (rshape.size() > 1 && rshape[-1] == 1) rshape = rshape.remove_end();

    if (lshape != rshape) TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    if (type() != rtype) TP_THROW(VERR_NOT_IMPLEMENTED_YET);

    ETensor y(nn(), rshape, VDataType::bool8);   // 32bit 정수형, 64비트 정수형 numpy 참고해 추후 결정

    int64 ndat = rshape.valid_size();

    unsigned char* pDst = y.bool_ptr();

    if (rtype == VDataType::bool8) {
        unsigned char* pl = bool_ptr();
        unsigned char* pr = rhs.bool_ptr();

        for (int64 n = 0; n < ndat; n++) {
            pDst[n] = pl[n] & pr[n];
        }
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }

    return y;
}

ETensor ETensor::operator |(ETensor rhs) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    // 예제에 나타난 산술연산은 서버 엔진 기능을 호출하지 않고 바로 처리한다.
    // 이는 예제에 표현되어 있음은 numpy 등을 이용해 게산한다는 의미를 갖기 때문이먀
    // 클라이언트 단의 연산이 가능하도록 데이터 다운로드 상태를 확인해야 하기 때문이기도 하다.

    VShape lshape = shape();
    VShape rshape = rhs.shape();

    VDataType rtype = rhs.type();

    while (lshape.size() > 1 && lshape[-1] == 1) lshape = lshape.remove_end();
    while (rshape.size() > 1 && rshape[-1] == 1) rshape = rshape.remove_end();

    if (lshape != rshape) TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    if (type() != rtype) TP_THROW(VERR_NOT_IMPLEMENTED_YET);

    ETensor y(nn(), rshape, VDataType::bool8);   // 32bit 정수형, 64비트 정수형 numpy 참고해 추후 결정

    int64 ndat = rshape.valid_size();

    unsigned char* pDst = y.bool_ptr();

    if (rtype == VDataType::bool8) {
        unsigned char* pl = bool_ptr();
        unsigned char* pr = rhs.bool_ptr();

        for (int64 n = 0; n < ndat; n++) {
            pDst[n] = pl[n] | pr[n];
        }
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }

    return y;
}

ETensor ETensor::operator*(float rhs) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    if (type() != VDataType::float32) {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }

    ETensor y(nn(), shape(), VDataType::float32);   // 32bit 정수형, 64비트 정수형 numpy 참고해 추후 결정

    int64 ndat = shape().valid_size();

    float* py = y.float_ptr();

    if (type() == VDataType::float32) {
        float* px = float_ptr();

        for (int64 n = 0; n < ndat; n++) {
            py[n] = px[n] * rhs;
        }
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }

    return y;
}

ETensor ETensor::operator/(float rhs) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    if (type() != VDataType::float32) {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }
    if (rhs == 0) {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }

    ETensor y(nn(), shape(), VDataType::float32);   // 32bit 정수형, 64비트 정수형 numpy 참고해 추후 결정

    int64 ndat = shape().valid_size();

    float* py = y.float_ptr();

    if (type() == VDataType::float32) {
        float* px = float_ptr();

        for (int64 n = 0; n < ndat; n++) {
            py[n] = px[n] / rhs;
        }
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }

    return y;
}

ETensor ETensor::operator+(float rhs) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    if (type() != VDataType::float32) {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }

    ETensor y(nn(), shape(), VDataType::float32);   // 32bit 정수형, 64비트 정수형 numpy 참고해 추후 결정

    int64 ndat = shape().valid_size();

    float* py = y.float_ptr();

    if (type() == VDataType::float32) {
        float* px = float_ptr();

        for (int64 n = 0; n < ndat; n++) {
            py[n] = px[n] + rhs;
        }
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }

    return y;
}

ETensor ETensor::operator-(float rhs) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    if (type() != VDataType::float32) {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }

    ETensor y(nn(), shape(), VDataType::float32);   // 32bit 정수형, 64비트 정수형 numpy 참고해 추후 결정

    int64 ndat = shape().valid_size();

    float* py = y.float_ptr();

    if (type() == VDataType::float32) {
        float* px = float_ptr();

        for (int64 n = 0; n < ndat; n++) {
            py[n] = px[n] - rhs;
        }
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }

    return y;
}

ETensor operator*(int value, ETensor x) {
    if (x.m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    
    VDataType type = x.type();

    ETensor y(x.nn(), x.shape(), type);   // 32bit 정수형, 64비트 정수형 numpy 참고해 추후 결정

    int64 ndat = x.shape().valid_size();

    if (type == VDataType::float32) {
        float* px = x.float_ptr();
        float* py = y.float_ptr();

        for (int64 n = 0; n < ndat; n++) {
            py[n] = px[n] * value;
        }
    }
    else if (type == VDataType::int64) {
        int64* px = x.int64_ptr();
        int64* py = y.int64_ptr();

        for (int64 n = 0; n < ndat; n++) {
            py[n] = px[n] * value;
        }
    }
    else if (type == VDataType::int32) {
        int* px = x.int_ptr();
        int* py = y.int_ptr();

        for (int64 n = 0; n < ndat; n++) {
            py[n] = px[n] * value;
        }
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }

    return y;
}

ETensor ETensor::operator*(int value) {
    ETensor x = *this;

    if (x.m_core == NULL) TP_THROW(VERR_INVALID_CORE);

    VDataType type = x.type();

    ETensor y(x.nn(), x.shape(), type);   // 32bit 정수형, 64비트 정수형 numpy 참고해 추후 결정

    int64 ndat = x.shape().valid_size();

    if (type == VDataType::float32) {
        float* px = x.float_ptr();
        float* py = y.float_ptr();

        for (int64 n = 0; n < ndat; n++) {
            py[n] = px[n] * value;
        }
    }
    else if (type == VDataType::int64) {
        int64* px = x.int64_ptr();
        int64* py = y.int64_ptr();

        for (int64 n = 0; n < ndat; n++) {
            py[n] = px[n] * value;
        }
    }
    else if (type == VDataType::int32) {
        int* px = x.int_ptr();
        int* py = y.int_ptr();

        for (int64 n = 0; n < ndat; n++) {
            py[n] = px[n] * value;
        }
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }

    return y;
}

ETensor ETensor::operator/(int value) {
    ETensor x = *this;

    if (x.m_core == NULL) TP_THROW(VERR_INVALID_CORE);

    VDataType type = x.type();

    ETensor y(x.nn(), x.shape(), type);   // 32bit 정수형, 64비트 정수형 numpy 참고해 추후 결정

    int64 ndat = x.shape().valid_size();

    if (type == VDataType::float32) {
        float* px = x.float_ptr();
        float* py = y.float_ptr();

        for (int64 n = 0; n < ndat; n++) {
            py[n] = px[n] / value;
        }
    }
    else if (type == VDataType::int64) {
        int64* px = x.int64_ptr();
        int64* py = y.int64_ptr();

        for (int64 n = 0; n < ndat; n++) {
            py[n] = px[n] / value;
        }
    }
    else if (type == VDataType::int32) {
        int* px = x.int_ptr();
        int* py = y.int_ptr();

        for (int64 n = 0; n < ndat; n++) {
            py[n] = px[n] / value;
        }
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }

    return y;
}

ETensor ETensor::operator+(int value) {
    ETensor x = *this;

    if (x.m_core == NULL) TP_THROW(VERR_INVALID_CORE);

    VDataType type = x.type();

    ETensor y(x.nn(), x.shape(), type);   // 32bit 정수형, 64비트 정수형 numpy 참고해 추후 결정

    int64 ndat = x.shape().valid_size();

    if (type == VDataType::float32) {
        float* px = x.float_ptr();
        float* py = y.float_ptr();

        for (int64 n = 0; n < ndat; n++) {
            py[n] = px[n] + value;
        }
    }
    else if (type == VDataType::int64) {
        int64* px = x.int64_ptr();
        int64* py = y.int64_ptr();

        for (int64 n = 0; n < ndat; n++) {
            py[n] = px[n] + value;
        }
    }
    else if (type == VDataType::int32) {
        int* px = x.int_ptr();
        int* py = y.int_ptr();

        for (int64 n = 0; n < ndat; n++) {
            py[n] = px[n] + value;
        }
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }

    return y;
}

ETensor ETensor::operator-(int value) {
    ETensor x = *this;

    if (x.m_core == NULL) TP_THROW(VERR_INVALID_CORE);

    VDataType type = x.type();

    ETensor y(x.nn(), x.shape(), type);   // 32bit 정수형, 64비트 정수형 numpy 참고해 추후 결정

    int64 ndat = x.shape().valid_size();

    if (type == VDataType::float32) {
        float* px = x.float_ptr();
        float* py = y.float_ptr();

        for (int64 n = 0; n < ndat; n++) {
            py[n] = px[n] - value;
        }
    }
    else if (type == VDataType::int64) {
        int64* px = x.int64_ptr();
        int64* py = y.int64_ptr();

        for (int64 n = 0; n < ndat; n++) {
            py[n] = px[n] - value;
        }
    }
    else if (type == VDataType::int32) {
        int* px = x.int_ptr();
        int* py = y.int_ptr();

        for (int64 n = 0; n < ndat; n++) {
            py[n] = px[n] - value;
        }
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }

    return y;
}

ETensor ETensor::operator*(int64 value) {
    ETensor x = *this;

    if (x.m_core == NULL) TP_THROW(VERR_INVALID_CORE);

    VDataType type = x.type();

    ETensor y(x.nn(), x.shape(), type);   // 32bit 정수형, 64비트 정수형 numpy 참고해 추후 결정

    int64 ndat = x.shape().valid_size();

    if (type == VDataType::float32) {
        float* px = x.float_ptr();
        float* py = y.float_ptr();

        for (int64 n = 0; n < ndat; n++) {
            py[n] = px[n] * value;
        }
    }
    else if (type == VDataType::int64) {
        int64* px = x.int64_ptr();
        int64* py = y.int64_ptr();

        for (int64 n = 0; n < ndat; n++) {
            py[n] = px[n] * value;
        }
    }
    else if (type == VDataType::int32) {
        int* px = x.int_ptr();
        int* py = y.int_ptr();

        for (int64 n = 0; n < ndat; n++) {
            py[n] = px[n] * (int)value;
        }
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }

    return y;
}

ETensor ETensor::operator/(int64 value) {
    ETensor x = *this;

    if (x.m_core == NULL) TP_THROW(VERR_INVALID_CORE);

    VDataType type = x.type();

    ETensor y(x.nn(), x.shape(), type);   // 32bit 정수형, 64비트 정수형 numpy 참고해 추후 결정

    int64 ndat = x.shape().valid_size();

    if (type == VDataType::float32) {
        float* px = x.float_ptr();
        float* py = y.float_ptr();

        for (int64 n = 0; n < ndat; n++) {
            py[n] = px[n] / value;
        }
    }
    else if (type == VDataType::int64) {
        int64* px = x.int64_ptr();
        int64* py = y.int64_ptr();

        for (int64 n = 0; n < ndat; n++) {
            py[n] = px[n] / value;
        }
    }
    else if (type == VDataType::int32) {
        int* px = x.int_ptr();
        int* py = y.int_ptr();

        for (int64 n = 0; n < ndat; n++) {
            py[n] = px[n] / (int)value;
        }
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }

    return y;
}

ETensor ETensor::operator+(int64 value) {
    ETensor x = *this;

    if (x.m_core == NULL) TP_THROW(VERR_INVALID_CORE);

    VDataType type = x.type();

    ETensor y(x.nn(), x.shape(), type);   // 32bit 정수형, 64비트 정수형 numpy 참고해 추후 결정

    int64 ndat = x.shape().valid_size();

    if (type == VDataType::float32) {
        float* px = x.float_ptr();
        float* py = y.float_ptr();

        for (int64 n = 0; n < ndat; n++) {
            py[n] = px[n] + value;
        }
    }
    else if (type == VDataType::int64) {
        int64* px = x.int64_ptr();
        int64* py = y.int64_ptr();

        for (int64 n = 0; n < ndat; n++) {
            py[n] = px[n] + value;
        }
    }
    else if (type == VDataType::int32) {
        int* px = x.int_ptr();
        int* py = y.int_ptr();

        for (int64 n = 0; n < ndat; n++) {
            py[n] = px[n] + (int)value;
        }
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }

    return y;
}

ETensor ETensor::operator-(int64 value) {
    ETensor x = *this;

    if (x.m_core == NULL) TP_THROW(VERR_INVALID_CORE);

    VDataType type = x.type();

    ETensor y(x.nn(), x.shape(), type);   // 32bit 정수형, 64비트 정수형 numpy 참고해 추후 결정

    int64 ndat = x.shape().valid_size();

    if (type == VDataType::float32) {
        float* px = x.float_ptr();
        float* py = y.float_ptr();

        for (int64 n = 0; n < ndat; n++) {
            py[n] = px[n] - value;
        }
    }
    else if (type == VDataType::int64) {
        int64* px = x.int64_ptr();
        int64* py = y.int64_ptr();

        for (int64 n = 0; n < ndat; n++) {
            py[n] = px[n] - value;
        }
    }
    else if (type == VDataType::int32) {
        int* px = x.int_ptr();
        int* py = y.int_ptr();

        for (int64 n = 0; n < ndat; n++) {
            py[n] = px[n] - (int)value;
        }
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }

    return y;
}

/*
ETensor ETensor::operator >(double value) {
    if (m_core == NULL) TP_THROW(VERR_UNDEFINED);
    return *this > (float)value;
}
*/

void ETensor::upload(bool force) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    if (m_core->m_hEngineHandle == 0) {
        m_core->m_needToClose = true;
        m_core->m_hEngineHandle = nn().getApiConn()->Tensor_create(__FILE__, __LINE__);
    }
    else if (!force && !m_core->m_needToUpload) {
        return;
    }

    nn().getApiConn()->Tensor_setFeature(m_core->m_hEngineHandle, m_core->m_shape, m_core->m_type, m_core->m_nDevice, __FILE__, __LINE__);

    int64 size = m_core->m_data.byteSize();

    if (size > 0) {
        nn().getApiConn()->Tensor_uploadData(m_core->m_hEngineHandle, m_core->m_data.void_ptr(), size, __FILE__, __LINE__);
    }

    //m_core->m_needToUpload = false;
}

ETensor ETensor::toDevice(int device) {
    if (m_core->m_nDevice == device) return *this;

    m_core->m_nDevice = device;
    
    VHTensor hDevTensor = nn().getApiConn()->Tensor_toDevice(m_core->m_hEngineHandle, device, __FILE__, __LINE__);

    ETensor devTensor(nn(), hDevTensor);

    VShape shape;
    VDataType dataType;
    int nDevice;

    nn().getApiConn()->Tensor_getFeature(hDevTensor, &shape, &dataType, &nDevice, __FILE__, __LINE__);

    devTensor.m_core->m_shape = shape;
    devTensor.m_core->m_type = dataType;
    devTensor.m_core->m_nDevice = nDevice;

    return devTensor;
}

void ETensor::backward() {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    if (m_core->m_hEngineHandle == 0) TP_THROW(VERR_UNDEFINED);
    nn().getApiConn()->Tensor_backward(m_core->m_hEngineHandle, __FILE__, __LINE__);
}

void ETensor::backwardWithGradient(ETensor grad) {
    if (m_core == NULL) TP_THROW(VERR_UNDEFINED);
    if (m_core->m_hEngineHandle == 0) TP_THROW(VERR_UNDEFINED);

    grad.upload();
    grad = grad.toDevice(device());

    nn().getApiConn()->Tensor_backwardWithGradient(m_core->m_hEngineHandle, (VHTensor)grad, __FILE__, __LINE__);
}

EScalar ETensor::item() {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    EScalar scalar(nn(), *this, m_core->m_fetchFloatScalar());
    return scalar;
}

int64 ETensor::len() {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    if (m_core->m_shape.size() == 0) TP_THROW(VERR_UNDEFINED);
    return m_core->m_shape[0];
}

int64 ETensor::size() {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    if (m_core->m_shape.size() == 0) return 0;
    return m_core->m_shape.valid_size();
}

VIntList ETensor::findElement(VValue element) {
    TP_THROW(VERR_NOT_IMPLEMENTED_YET);
}

string ETensor::desc() {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    void* pHostBuf = void_ptr();

    VShape xshape = shape();

    if (xshape.size() == 1) {
        string vector_desc = "[";

        if (xshape[0] > 0) vector_desc += m_getElementDesc(pHostBuf, 0);

        for (int64 n = 1; n < xshape[0]; n++) vector_desc += "," + m_getElementDesc(pHostBuf, n);

        vector_desc += "]";

        return vector_desc;
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }
}

void ETensor::m_dump(VShape shape, void* pHostBuf, int64 nth, int indent, bool full) {
    VShape tshape = shape.remove_head();

    int64 ax_size = shape[0];
    int64 child_size = tshape.valid_size();

    if (shape.size() == 0) return;
    else if (shape.size() == 1) {
        printf("[");
        if (full || ax_size <= 7) {
            for (int64 n = 0; n < ax_size; n++) m_dumpNthElement(pHostBuf, nth++, n > 0);
        }
        else {
            for (int64 n = 0; n < 3 ; n++) m_dumpNthElement(pHostBuf, nth++, n > 0);
            nth += ax_size - 6;
            printf(" ...");
            for (int64 n = 0; n < 3; n++) m_dumpNthElement(pHostBuf, nth++);
        }
        printf("]");
    }
    else {
        bool needNewLine = shape.valid_size() > 12;
        printf("[");
        if (full || ax_size <= 5) {
            for (int64 n = 0; n < ax_size; n++) {
                if (n > 0 && needNewLine) {
                    printf("\n%*s", indent, "");
                }
                m_dump(tshape, pHostBuf, nth, indent+2, full);
                nth += child_size;
            }
        }
        else {
            m_dump(tshape, pHostBuf, nth, indent + 2, false);
            nth += child_size;
            if (needNewLine) printf("\n%*s", indent, "");
            m_dump(tshape, pHostBuf, nth, indent + 2, false);
            if (needNewLine) printf("\n%*s", indent, "");
            nth += child_size;

            nth += (ax_size - 4) * child_size;
            printf("...");
            
            if (needNewLine) printf("\n%*s", indent, "");
            m_dump(tshape, pHostBuf, nth, indent + 2, false);
            nth += child_size;
            if (needNewLine) printf("\n%*s", indent, "");
            m_dump(tshape, pHostBuf, nth, indent + 2, false);
        }
        printf("]");
    }
}

string ETensor::m_dump_to_str(VShape shape, void* pHostBuf, int64 nth, int indent, bool full) {
    string outstr = "";

    VShape tshape = shape.remove_head();

    /*
    if (tshape.size() == 0) {
        m_dumpNthElement(pHostBuf, nth, false);
        return;
    }
    */

    int64 ax_size = shape[0];
    int64 child_size = tshape.valid_size();

    if (shape.size() == 0) {
        return outstr;
    }
    else if (shape.size() == 1) {
        //print("[");
        outstr += string("[");
        if (full || ax_size <= 7) {
            for (int64 n = 0; n < ax_size; n++) outstr += m_dumpNthElement_to_str(pHostBuf, nth++, n > 0);
        }
        else {
            for (int64 n = 0; n < 3 ; n++) outstr += m_dumpNthElement_to_str(pHostBuf, nth++, n > 0);
            nth += ax_size - 6;
            outstr += string(" ...");
            for (int64 n = 0; n < 3; n++) outstr += m_dumpNthElement_to_str(pHostBuf, nth++);
        }
        outstr += string("]");
    }
    else {
        bool needNewLine = shape.valid_size() > 12;
        outstr += string("[");
        if (full || ax_size <= 5) {
            for (int64 n = 0; n < ax_size; n++) {
                if (n > 0 && needNewLine) {
                    char szBuff[4096] = {0, };
                    sprintf(szBuff, "%*s", indent, "");
                    outstr += string(szBuff);
                }
                outstr += m_dump_to_str(tshape, pHostBuf, nth, indent+2, true);
                nth += child_size;
            }
        }
        else {
            outstr += m_dump_to_str(tshape, pHostBuf, nth, indent + 2, false);
            nth += child_size;
            if (needNewLine) {
                char szBuff[4096] = {0, };
                sprintf(szBuff, "%*s", indent, "");
                outstr += string(szBuff);
            }
            outstr += m_dump_to_str(tshape, pHostBuf, nth, indent + 2, false);
            if (needNewLine) {
                char szBuff[4096] = {0, };
                sprintf(szBuff, "%*s", indent, "");
                outstr += string(szBuff);
            }
            nth += child_size;

            nth += (ax_size - 4) * child_size;
            outstr += string("...");
            
            if (needNewLine) {
                char szBuff[4096] = {0, };
                sprintf(szBuff, "%*s", indent, "");
                outstr += string(szBuff);
            }
            outstr += m_dump_to_str(tshape, pHostBuf, nth, indent + 2, false);
            nth += child_size;
            if (needNewLine) {
                char szBuff[4096] = {0, };
                sprintf(szBuff, "%*s", indent, "");
                outstr += string(szBuff);
            }
            outstr += m_dump_to_str(tshape, pHostBuf, nth, indent + 2, false);
        }
        outstr += string("]");
    }

    return outstr;
}

void ETensor::dump(string title, bool full) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    //printf("T#%s%s:%s", title.c_str(), shape().desc().c_str(), VDataTypeName(type()).c_str());
    printf("T#%d:%s%s:%s(dev:%d)", nn().getEngineObjId(m_core), title.c_str(), shape().desc().c_str(), VDataTypeName(type()).c_str(), device());

    if (shape().valid_size() > 0) {
        void* pHostBuf = void_ptr();

        if (shape().valid_size() > 12) printf("\n    ");

        m_dump(shape(), pHostBuf, 0, 4, full);
    }

    printf("\n");
}

string ETensor::get_dump_str(string title, bool full) {
    if (m_core == NULL) TP_THROW(VERR_UNDEFINED);

    char szBuff[4096] = {0, };

    string outstr = "";
    
    memset(szBuff, 0, sizeof(szBuff));
    sprintf(szBuff, "T#%s%s:%s", title.c_str(), shape().desc().c_str(), VDataTypeName(type()).c_str());
    outstr += string(szBuff);

    void* pHostBuf = void_ptr();

    if (shape().valid_size() > 12) {
        memset(szBuff, 0, sizeof(szBuff));
        sprintf(szBuff, "    ");
        outstr += string(szBuff);
    }

    outstr += m_dump_to_str(shape(), pHostBuf, 0, 4, full);

    outstr += string("");

    return outstr;
}

void ETensor::free_dump_str(const char* pData) {
    printf("depreciated %s() in %s:%d\n", __func__, __FILE__, __LINE__);
    //free(pData);
}

void ETensor::m_dumpElement(void* pHostBuf, int64 n) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    print("%s ", m_getElementDesc(pHostBuf, n).c_str());
}

void ETensor::m_dumpNthElement(void* pHostBuf, int64 n, bool bSpaceOnLeft) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    if (bSpaceOnLeft) printf(" ");
    printf("%s", m_getElementDesc(pHostBuf, n).c_str());
}

string ETensor::m_dumpNthElement_to_str(void* pHostBuf, int64 n, bool bSpaceOnLeft) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);

    string outstr = "";

    if (bSpaceOnLeft) outstr += string(" ");

    outstr += m_getElementDesc(pHostBuf, n);

    return outstr;
}

string ETensor::m_getElementDesc(void* pHostBuf, int64 n) {
    char buffer[128];

    switch (type()) {
    case VDataType::float32:
        snprintf(buffer, 128, "%f", ((float*)pHostBuf)[n]);
        break;
    case VDataType::int32:
        snprintf(buffer, 128, "%d", ((int*)pHostBuf)[n]);
        break;
    case VDataType::bool8:
        snprintf(buffer, 128, "%s", ((unsigned char*)pHostBuf)[n] ? "True" : "False");
        break;
    case VDataType::uint8:
        snprintf(buffer, 128, "%d", ((unsigned char*)pHostBuf)[n]);
        break;
    case VDataType::int64:
        snprintf(buffer, 128, "%lld", ((int64*)pHostBuf)[n]);
        break;
    default:
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }

    return string(buffer);
}

ETensor ETensor::operator[](int64 index) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    ETensor tensor(nn(), shape().remove_nth(0), type());

    tensor.m_core->m_fetchIdxRows(*this, &index, 1);

    return tensor;
}

ETensor ETensor::operator[](ETensor index) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);

    ETensor x = *this;
    
    VDataType x_type = x.type();
    VDataType idx_type = index.type();

    if (x_type != VDataType::int32 && x_type != VDataType::int64 && x_type != VDataType::float32) TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    if (idx_type != VDataType::int32 && idx_type != VDataType::int64) TP_THROW(VERR_NOT_IMPLEMENTED_YET);

    VShape xshape = x.shape();
    VShape ishape = index.shape();
    VShape yshape = ishape.append(xshape.remove_head());

    int64 nidx = ishape[0];
    int64 nnom = xshape[0];
    int64 nvec = xshape.valid_size() / nnom;

    ETensor y(x.nn(), yshape, x_type);

    for (int64 ni = 0; ni < nidx; ni++) {
        int64 nr = index[ni];
        for (int64 nv = 0; nv < nvec; nv++) {
            int64 xpos = nr * nvec + nv;
            int64 ypos = ni * nvec + nv;
            y.setElement(ypos, x.getElement(xpos));
        }
    }

    return y;
}

ETensor ETensor::argmax(int64 axis) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    // 예제에 나타난 산술연산은 서버 엔진 기능을 호출하지 않고 바로 처리한다.
    // 이는 예제에 표현되어 있음은 numpy 등을 이용해 계산한다는 의미를 갖기 때문이먀
    // 클라이언트 단의 연산이 가능하도록 데이터 다운로드 상태를 확인해야 하기 때문이기도 하다.

    //VShape shape = m_core->m_shape.replace_nth(axis, 1);
    VShape shape = m_core->m_shape.remove_nth(axis);
    ETensor tensor(nn(), shape, VDataType::int32);   // 32bit 정수형, 64비트 정수형 numpy 참고해 추후 결정
    tensor.m_core->m_argmax(*this, axis);
    return tensor;
}

ETensor ETensor::evaluate(string expression, VList args) {
    TP_THROW(VERR_NOT_IMPLEMENTED_YET);
}

ETensor ETensor::linspace(float from, float to, int count, VDict kwArgs) {
    TP_THROW(VERR_NOT_IMPLEMENTED_YET);
}

ETensor ETensor::sin() {
    TP_THROW(VERR_NOT_IMPLEMENTED_YET);
}

ETensor ETensor::randh(VShape shape, VDict kwArgs) {
    TP_THROW(VERR_NOT_IMPLEMENTED_YET);
}

ETensor ETensor::sum() {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    // 예제에 나타난 산술연산은 서버 엔진 기능을 호출하지 않고 바로 처리한다.
    // 이는 예제에 표현되어 있음은 numpy 등을 이용해 계산한다는 의미를 갖기 때문이먀
    // 클라이언트 단의 연산이 가능하도록 데이터 다운로드 상태를 확인해야 하기 때문이기도 하다.

    ETensor tensor(nn(), { 1 }, type());   // 32bit 정수형, 64비트 정수형 numpy 참고해 추후 결정
    tensor.m_core->m_sum(*this);
    return tensor;
}

ETensor ETensor::mean() {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);

    ETensor tensor(nn(), { 1 }, type());   // 32bit 정수형, 64비트 정수형 numpy 참고해 추후 결정
    tensor.m_core->m_mean(*this);
    return tensor;
}

ETensor abs(ETensor term) {
    return term.abs();
}

ETensor ETensor::abs() {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    ETensor tensor(nn(), shape(), type());   // 32bit 정수형, 64비트 정수형 numpy 참고해 추후 결정
    tensor.m_core->m_abs(*this);
    return tensor;
}

ETensor ETensor::sigmoid() {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    ETensor tensor(nn(), shape(), type());   // 32bit 정수형, 64비트 정수형 numpy 참고해 추후 결정
    tensor.m_core->m_sigmoid(*this);
    return tensor;
}

ETensor ETensor::square() {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    ETensor tensor(nn(), shape(), type());   // 32bit 정수형, 64비트 정수형 numpy 참고해 추후 결정
    tensor.m_core->m_square(*this);
    return tensor;
}

ETensor ETensor::transpose() {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    if (shape().size() != 2) TP_THROW2(VERR_SHAPE_TENSOR, "transpose");
    ETensor tensor(nn(), { shape()[1], shape()[0] }, type());
    tensor.m_core->m_transpose(*this);
    return tensor;
}

ETensor ETensor::transpose(VList axes) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    if (shape().size() != axes.size()) TP_THROW2(VERR_SHAPE_TENSOR, "transpose");

    TpCuda cuda;
    return cuda.transpose(nn(), *this, axes);
}

ETensor max(ETensor lhs, ETensor rhs) {
    return lhs.max(rhs);
}

ETensor ETensor::max(ETensor rhs) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    ETensor tensor(nn(), shape(), type());   // 32bit 정수형, 64비트 정수형 numpy 참고해 추후 결정
    tensor.m_core->m_max(*this, rhs);
    return tensor;
}

ETensor ETensor::operator*(ETensor rhs) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    VDataType result_type = m_op_result_type(type(), rhs.type());
    ETensor tensor(nn(), shape(), result_type);
    if (shape() != rhs.shape()) TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    tensor.m_core->m_mult(*this, rhs);
    return tensor;
}

ETensor ETensor::operator/(ETensor rhs) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    VDataType result_type = m_op_result_type(type(), rhs.type());
    ETensor tensor(nn(), shape(), result_type);
    if (shape() != rhs.shape()) TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    tensor.m_core->m_div(*this, rhs);
    return tensor;
}

ETensor ETensor::operator-(ETensor rhs) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    VDataType result_type = m_op_result_type(type(), rhs.type());
    ETensor tensor(nn(), shape(), result_type);
    tensor.m_core->m_subtract(*this, rhs);
    return tensor;
}

ETensor ETensor::operator+(ETensor rhs) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    VDataType result_type = m_op_result_type(type(), rhs.type());
    ETensor tensor(nn(), shape(), result_type);
    if (shape() != rhs.shape()) TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    tensor.m_core->m_add(*this, rhs);
    return tensor;
}

ETensor::operator int() const {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    if (m_core->m_shape.valid_size() != 1) TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    return m_core->m_fetchIntScalar();
}

int64 ETensor::to_int64() const {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    if (m_core->m_shape.valid_size() != 1) TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    return m_core->m_fetchInt64Scalar();
}

ETensor::operator float() const {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    if (m_core->m_shape.valid_size() != 1) TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    return m_core->m_fetchFloatScalar();
}

void ETensor::setDebugName(string name) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    m_core->m_debugName = name;
}

int64 ETensor::byteSize() {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    return m_core->m_data.byteSize();
}

void ETensor::copyPartialData(ETensor src) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    if (byteSize() < src.byteSize()) TP_THROW(VERR_SIZE_TENSOR);

    void* pDest = void_ptr();
    void* pSrc = src.void_ptr();

    memcpy(pDest, pSrc, src.byteSize());
}

void ETensor::copyRowFrom(int64 destRow, ETensor src, int64 srcRow) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    int64 row_size = byteSize() / shape()[0];
    
    if (row_size != src.byteSize() / src.shape()[0]) TP_THROW2(VERR_SHAPE_TENSOR, "copyRowFrom");
    
    if (destRow < 0 || destRow >= shape()[0]) TP_THROW2(VERR_SHAPE_TENSOR, "copyRowFrom");
    if (srcRow < 0 || srcRow >= src.shape()[0]) TP_THROW2(VERR_SHAPE_TENSOR, "copyRowFrom");

    unsigned char* pDest = (unsigned char* )void_ptr() + row_size * destRow;
    unsigned char* pSrc = (unsigned char* )src.void_ptr() + row_size * srcRow;

    memcpy(pDest, pSrc, row_size);
}

void ETensor::shift_timestep_to_right(ETensor src, int64 steps) {
    TP_THROW(VERR_INVALID_CORE);
}

void ETensor::dump_arr_feat(int nth, string title) {
    ETensor x = *this;
    float sum = 0, dsqsum = 0;
    int64 minpos = 0, maxpos = 0;

    int64 size = x.shape().valid_size();

    print("%d\t%s\t%s\tdevice:%d",
        nth, title.c_str(), x.shape().desc().c_str(), x.device());

    if (x.type() == VDataType::float32) {
        float* pBuffer = x.float_ptr();
        float first, last, min, max, avg, std;

        first = min = max = pBuffer[0];
        last = pBuffer[size - 1];
        for (int64 n = 0; n < size; n++) {
            if (pBuffer[n] > max) {
                max = pBuffer[n];
                maxpos = n;
            }
            if (pBuffer[n] < min) {
                min = pBuffer[n];
                minpos = n;
            }
            sum += pBuffer[n];
        }

        avg = sum / (float)size;

        for (int64 n = 0; n < size; n++) {
            dsqsum += (pBuffer[n] - avg) * (pBuffer[n] - avg);
        }

        avg = sum / (float)size;
        std = ::sqrtf(dsqsum / (float)size + 1e-10f);

        print("\tfirst:%g\tlast:%g\tmin:%g(%lld)\tmax:%g(%lld)\tavg:%g\tstd:%g",
            first, last, min, minpos, max, maxpos, avg, std);

        int histogram[11];
        int nout = 0;

        memset(histogram, 0, 11 * sizeof(int));

        if (min < max) {
            for (int64 n = 0; n < size; n++) {
                if (pBuffer[n] == min) histogram[0]++;
                else if (pBuffer[n] == max) histogram[10]++;
                else {
                    int nth = (int)((pBuffer[n] - min) * 9 / (max - min));
                    if (nth >= 0 && nth < 10) histogram[nth]++;
                    else nout++;
                }
            }
        }
        else {
            histogram[0] = (int)size;
        }
        print("\t\t\tHistogram:");
        for (int64 n = 0; n < 10; n++) {
            print(" %d", histogram[n]);
        }
        print(" [out: %d]", nout);
    }
    else if (x.type() == VDataType::int32) {
        int* pBuffer = x.int_ptr();
        int first, last, min, max;
        float avg, std;

        first = min = max = pBuffer[0];
        last = pBuffer[size - 1];
        for (int64 n = 0; n < size; n++) {
            if (pBuffer[n] > max) {
                max = pBuffer[n];
                maxpos = n;
            }
            if (pBuffer[n] < min) {
                min = pBuffer[n];
                minpos = n;
            }
            sum += pBuffer[n];
        }

        avg = sum / (float)size;

        for (int64 n = 0; n < size; n++) {
            dsqsum += (pBuffer[n] - avg) * (pBuffer[n] - avg);
        }

        std = ::sqrtf(dsqsum / size + 1e-10f);

        print("\tfirst:%d\tlast:%d\tmin:%d(%lld)\tmax:%d(%lld)\tavg:%g\tstd:%g",
            first, last, min, minpos, max, maxpos, avg, std);

        int histogram[11];
        int nout = 0;

        memset(histogram, 0, 11 * sizeof(int));

        if (min < max) {
            for (int64 n = 0; n < size; n++) {
                if (pBuffer[n] == min) histogram[0]++;
                else if (pBuffer[n] == max) histogram[10]++;
                else {
                    int nth = (int)((pBuffer[n] - min) * 9 / (max - min));
                    if (nth >= 0 && nth < 10) histogram[nth]++;
                    else nout++;
                }
            }
        }
        else {
            histogram[0] = (int)size;
        }
        print("\t\t\tHistogram:");
        for (int64 n = 0; n < 10; n++) {
            print(" %d", histogram[n]);
        }
        print(" [out: %d]", nout);
    }
}

VDataType ETensor::m_op_result_type(VDataType type1, VDataType type2) {
    if (type1 != VDataType::float32 && type1 == VDataType::int64 && type1 == VDataType::int32) TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    if (type2 != VDataType::float32 && type2 == VDataType::int64 && type2 == VDataType::int32) TP_THROW(VERR_NOT_IMPLEMENTED_YET);

    if (type1 == VDataType::float32 || type2 == VDataType::float32) return VDataType::float32;
    if (type1 == VDataType::int64 || type2 == VDataType::int64) return VDataType::int64;
    return VDataType::int32;
}

//-----------------------------------------------------------------------------------------------------
// Core part

void ETensorCore::m_setup() {
    m_needToClose = false;
    m_needToUpload = true;
    m_nDevice = -1;
}

void ETensorCore::m_delete() {
}

void ETensorCore::m_createData(void* pData) {
    int64 size = m_shape.valid_size() * TpUtils::byte_size(m_type);

    m_data = ETensorData(m_nn, size);

    if (pData) m_data.copyFrom(pData);
}

void ETensorCore::m_createData(FILE* fin, VDataType loadType) {
    int64 count = m_shape.valid_size();
    int64 size = count * TpUtils::byte_size(m_type);

    m_data = ETensorData(m_nn, size);

    if (loadType == m_type) {
        fread(m_data.void_ptr(), TpUtils::byte_size(m_type), count, fin);
    }
    else if (loadType == VDataType::float64 && m_type == VDataType::float32) {
        double buffer;
        float* pDest = m_data.float_ptr();
        for (int64 n = 0; n < count; n++) {
            fread(&buffer, 8, 1, fin);
            pDest[n] = (float)buffer;
        }
    }
    else {
        TP_THROW(VERR_TENSOR_DATATYPE);
    }
}

void ETensorCore::m_downloadData(VHTensor hTensor) {
    if (!m_data.isValid()) {
        int64 size = m_shape.valid_size() * TpUtils::byte_size(m_type);
        if (size == 0) return;
        m_data = ETensorData(m_nn, size);
    }

    if (hTensor) {
        m_data.downloadFrom(hTensor);
    }
}

void ETensorCore::m_downloadData(ENN nn, VHTensor hTensor) {
    if (!m_data.isValid()) {
        int64 size = m_shape.valid_size() * TpUtils::byte_size(m_type);
        m_data = ETensorData(m_nn, size);
    }

    if (hTensor) {
        m_data.downloadFrom(nn, hTensor);
    }
}

void ETensorCore::m_copyData(void* pData) {
    if (!m_data.isValid()) TP_THROW(VERR_UNDEFINED); 

    if (pData) m_data.copyFrom(pData);
}

void ETensorCore::m_reset() {
    if (m_data.isValid()) {
        m_data.reset();
    }
}

void ETensorCore::m_to_type(ETensor src, string option) {
    if (m_shape.valid_size() != src.shape().valid_size()) TP_THROW(VERR_SIZE_TENSOR);

    int count = (int)m_shape.valid_size();

    if (m_type == VDataType::float32) {
        float* pDst = m_data.float_ptr();

        if (src.type() == VDataType::uint8) {
            unsigned char* pSrc = src.uchar_ptr();
            if (option == "unit") {
                for (int n = 0; n < count; n++) pDst[n] = (pSrc[n] - 127.5f) / 127.5f;
            }
            else if (option == "posunit") {
                for (int n = 0; n < count; n++) pDst[n] = pSrc[n] / 255.0f;
            }
            else {
                for (int n = 0; n < count; n++) pDst[n] = (float)pSrc[n];
            }
        }
        else if (src.type() == VDataType::bool8) {
            unsigned char* pSrc = src.bool_ptr();
            for (int n = 0; n < count; n++) pDst[n] = (pSrc[n] != 0) ? 1.0f : 0.0f;
        }
        else if (src.type() == VDataType::int32) {
            int* pSrc = src.int_ptr();
            for (int n = 0; n < count; n++) pDst[n] = (float)pSrc[n];
        }
        else {
            TP_THROW(VERR_NOT_IMPLEMENTED_YET);
        }
    }
    else if (m_type == VDataType::int32) {
        int* pDst = m_data.int_ptr();
        if (src.type() == VDataType::uint8) {
            unsigned char* pSrc = src.uchar_ptr();
            for (int n = 0; n < count; n++) {
                int nVal = (int)pSrc[n];
                if (nVal < 0 || nVal > 9) {
                    //print("pSrc[%d] = %d => %lld", n, pSrc[n], nVal);
                }
                pDst[n] = nVal;
            }
        }
        else if (src.type() == VDataType::bool8) {
            unsigned char* pSrc = src.bool_ptr();;
            for (int n = 0; n < count; n++) pDst[n] = (pSrc[n] != 0) ? 1 : 0;
        }
        else if (src.type() == VDataType::float32) {
            float* pSrc = src.float_ptr();;
            for (int n = 0; n < count; n++) pDst[n] = (int)pSrc[n];
        }
        else {
            TP_THROW(VERR_NOT_IMPLEMENTED_YET);
        }
    }
    else if (m_type == VDataType::int64) {
        int64* pDst = m_data.int64_ptr();
        if (src.type() == VDataType::uint8) {
            unsigned char* pSrc = src.uchar_ptr();
            for (int n = 0; n < count; n++) {
                int64 nVal = (int64)pSrc[n];
                if (nVal < 0 || nVal > 9) {
                    //print("pSrc[%d] = %d => %lld", n, pSrc[n], nVal);
                }
                pDst[n] = nVal;
            }
        }
        else if (src.type() == VDataType::bool8) {
            unsigned char* pSrc = src.bool_ptr();;
            for (int n = 0; n < count; n++) pDst[n] = (pSrc[n] != 0) ? 1 : 0;
        }
        else if (src.type() == VDataType::float32) {
            float* pSrc = src.float_ptr();;
            for (int n = 0; n < count; n++) pDst[n] = (int64)pSrc[n];
        }
        else {
            TP_THROW(VERR_NOT_IMPLEMENTED_YET);
        }
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }
}

int ETensorCore::m_fetchIntScalar() {
    if (m_shape.valid_size() != 1) TP_THROW(VERR_SIZE_TENSOR);

    if (m_type == VDataType::float32) {
        return (int)*m_data.int_ptr();
    }
    else if (m_type == VDataType::int32) {
        return (int)*m_data.int_ptr();
    }
    else if (m_type == VDataType::int64) {
        return (int)*m_data.int64_ptr();
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }
}

int64 ETensorCore::m_fetchInt64Scalar() {
    if (m_shape.valid_size() != 1) TP_THROW(VERR_SIZE_TENSOR);

    if (m_type == VDataType::float32) {
        return (int64)*m_data.int_ptr();
    }
    else if (m_type == VDataType::int32) {
        return (int64)*m_data.int_ptr();
    }
    else if (m_type == VDataType::int64) {
        return (int64)*m_data.int64_ptr();
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }
}

float ETensorCore::m_fetchFloatScalar() {
    if (m_shape.valid_size() != 1) TP_THROW(VERR_SIZE_TENSOR);

    if (!m_data.isValid()) {
        int64 size = m_shape.valid_size() * TpUtils::byte_size(m_type);
        m_data = ETensorData(m_nn, size);
        m_data.downloadFrom(m_hEngineHandle);
    }

    if (m_type == VDataType::float32) {
        return *m_data.float_ptr();
    }
    else if (m_type == VDataType::int32) {
        return (float)(*m_data.int_ptr());
    }
    else if (m_type == VDataType::int64) {
        return (float)(*m_data.int64_ptr());
    }
    else {
        TP_THROW(VERR_SIZE_TENSOR);
    }
}

void ETensorCore::m_fetchIdxRows(ETensor src, int64* pnMap, int64 batch_size) {
    int64 dat_count = m_shape.valid_size() / batch_size;

    if (batch_size != 1 && batch_size != m_shape[0]) TP_THROW2(VERR_SHAPE_TENSOR, "fetchIdxRows");
    if (src.shape().valid_size() / src.shape()[0] != dat_count) TP_THROW2(VERR_SHAPE_TENSOR, "fetchIdxRows");
    if (m_type != src.type()) TP_THROW(VERR_TENSOR_DATATYPE);

    int64 dat_size = dat_count * TpUtils::byte_size(m_type);

    // void* 대신 unsigned char* 타입을 사용하는 이유는 루프 내에서 포인터에 int64 더해주는 연산이 지원 안되기 때문임
    unsigned char* pDstBase = (unsigned char*)m_data.void_ptr();
    unsigned char* pSrcBase = (unsigned char*)src.void_ptr();

    for (int64 n = 0; n < batch_size; n++) {
        int64 temp = pnMap[n];
        unsigned char* pDst = pDstBase + n * dat_size;
        unsigned char* pSrc = pSrcBase + pnMap[n] * dat_size;

        memcpy(pDst, pSrc, dat_size);
    }
}

void ETensorCore::m_fetchIdxRows(ETensor src, int* pnMap, int64 batch_size) {
    int64 dat_count = m_shape.valid_size() / batch_size;

    if (batch_size != 1 && batch_size != m_shape[0]) TP_THROW2(VERR_SHAPE_TENSOR, "fetchIdxRows");
    if (src.shape().valid_size() / src.shape()[0] != dat_count) TP_THROW2(VERR_SHAPE_TENSOR, "fetchIdxRows");
    if (m_type != src.type()) TP_THROW(VERR_TENSOR_DATATYPE);

    int64 dat_size = dat_count * TpUtils::byte_size(m_type);

    // void* 대신 unsigned char* 타입을 사용하는 이유는 루프 내에서 포인터에 int64 더해주는 연산이 지원 안되기 때문임
    unsigned char* pDstBase = (unsigned char*)m_data.void_ptr();
    unsigned char* pSrcBase = (unsigned char*)src.void_ptr();

    for (int64 n = 0; n < batch_size; n++) {
        unsigned char* pDst = pDstBase + n * dat_size;
        unsigned char* pSrc = pSrcBase + pnMap[n] * dat_size;

        memcpy(pDst, pSrc, dat_size);
    }
}

void ETensorCore::m_copy_into_row(int64 nthRow, ETensor src) {
    int64 batch_size = m_shape[0];
    int64 dat_size = m_shape.valid_size() / batch_size;

    if (dat_size != src.shape().valid_size()) TP_THROW2(VERR_SHAPE_TENSOR, "copy_into_row");
    if (nthRow < 0 || nthRow >= batch_size) TP_THROW(VERR_UNDEFINED);
    if (m_type != src.type()) TP_THROW(VERR_TENSOR_DATATYPE);

    // void* 대신 unsigned char* 타입을 사용하는 이유는 루프 내에서 포인터에 int64 더해주는 연산이 지원 안되기 때문임
    unsigned char* pDst = (unsigned char*)m_data.void_ptr() + nthRow * src.byteSize();
    unsigned char* pSrc = (unsigned char*)src.void_ptr();

    memcpy(pDst, pSrc, src.byteSize());
}

void ETensorCore::m_argmax(ETensor opnd, int64 axis) {
    int64 ndat = m_shape.valid_size();
    int64 ntail = m_shape.tail_size(axis);
    int64 nhead = ndat / ntail;
    int64 nvec = opnd.shape()[axis];

    if (opnd.hasNoData()) opnd.downloadData();

    int* pDst = m_data.int_ptr();
    float* pSrc = opnd.float_ptr();

    for (int64 nh = 0; nh < nhead; nh++) {
        for (int64 nt = 0; nt < ntail; nt++) {
            int arg = 0;
            float max = -FLT_MAX;

            for (int64 m = 0; m < nvec; m++) {
                int64 idx = (nh * nvec + m) * ntail + nt;
                if (pSrc[idx] > max) {
                    arg = (int)m;
                    max = pSrc[idx];
                }
            }

            pDst[nh * ntail + nt] = arg;
        }
    }
}

void ETensorCore::m_sum(ETensor opnd) {
    int64 ndat = opnd.shape().valid_size();

    if (m_type == VDataType::float32) {
        float* pDst = m_data.float_ptr();
        float* pSrc = opnd.float_ptr();

        float sum = 0;

        for (int64 n = 0; n < ndat; n++) {
            sum += pSrc[n];
        }

        pDst[0] = sum;
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }
}

void ETensorCore::m_mean(ETensor opnd) {
    int64 ndat = opnd.shape().valid_size();

    if (m_type == VDataType::float32) {
        float* pDst = m_data.float_ptr();
        float* pSrc = opnd.float_ptr();

        float sum = 0;

        for (int64 n = 0; n < ndat; n++) {
            sum += pSrc[n];
        }

        pDst[0] = sum / ndat;
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }
}

void ETensorCore::m_abs(ETensor opnd) {
    int64 ndat = opnd.shape().valid_size();

    if (m_type == VDataType::float32) {
        float* pDst = m_data.float_ptr();
        float* pSrc = opnd.float_ptr();

        for (int64 n = 0; n < ndat; n++) {
            pDst[n] = ::fabsf(pSrc[n]);
        }
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }
}

void ETensorCore::m_sigmoid(ETensor opnd) {
    int64 ndat = opnd.shape().valid_size();

    if (m_type == VDataType::float32) {
        float* pDst = m_data.float_ptr();
        float* pSrc = opnd.float_ptr();

        for (int64 n = 0; n < ndat; n++) {
            float x = pSrc[n];
            float prob = (x > 0) ? (1.0f / (1.0f + ::expf(-x))) : (::expf(x) / (::expf(x) + 1.0f));
            pDst[n] = prob;
        }
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }
}

void ETensorCore::m_square(ETensor opnd) {
    int64 ndat = opnd.shape().valid_size();

    if (m_type == VDataType::float32) {
        float* pDst = m_data.float_ptr();
        float* pSrc = opnd.float_ptr();

        for (int64 n = 0; n < ndat; n++) {
            float x = pSrc[n];
            pDst[n] = x * x;
        }
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }
}

void ETensorCore::m_transpose(ETensor opnd) {
    int64 nrow = m_shape[0];
    int64 ncol = m_shape[1];

    if (m_type == VDataType::float32) {
        float* pDst = m_data.float_ptr();
        float* pSrc = opnd.float_ptr();

        for (int64 nr = 0; nr < nrow; nr++) {
            for (int64 nc = 0; nc < ncol; nc++) {
                pDst[nr * ncol + nc] = pSrc[nc * nrow + nr];
            }
        }
    }
    else {
        TP_THROW(VERR_TENSOR_DATATYPE);
    }
}

void ETensorCore::m_mult(ETensor opnd1, ETensor opnd2) {
    int64 ndat = opnd1.shape().valid_size();

    if (m_type == VDataType::float32) {
        float* pDst = m_data.float_ptr();

        for (int64 n = 0; n < ndat; n++) {
            float value = (float)opnd1.getElement(n) * (float)opnd2.getElement(n);
            pDst[n] = value;
        }
    }
    else if (m_type == VDataType::int64) {
        int64* pDst = m_data.int64_ptr();

        for (int64 n = 0; n < ndat; n++) {
            int64 value = (int64)opnd1.getElement(n) * (int64)opnd2.getElement(n);
            pDst[n] = value;
        }
    }
    else if (m_type == VDataType::int32) {
        int* pDst = m_data.int_ptr();

        for (int64 n = 0; n < ndat; n++) {
            int value = (int)opnd1.getElement(n) * (int)opnd2.getElement(n);
            pDst[n] = value;
        }
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }
}

void ETensorCore::m_div(ETensor opnd1, ETensor opnd2) {
    int64 ndat = opnd1.shape().valid_size();

    if (m_type == VDataType::float32) {
        float* pDst = m_data.float_ptr();

        for (int64 n = 0; n < ndat; n++) {
            float value = (float)opnd1.getElement(n) / (float)opnd2.getElement(n);
            pDst[n] = value;
        }
    }
    else if (m_type == VDataType::int64) {
        int64* pDst = m_data.int64_ptr();

        for (int64 n = 0; n < ndat; n++) {
            int64 value = (int64)opnd1.getElement(n) / (int64)opnd2.getElement(n);
            pDst[n] = value;
        }
    }
    else if (m_type == VDataType::int32) {
        int* pDst = m_data.int_ptr();

        for (int64 n = 0; n < ndat; n++) {
            int value = (int)opnd1.getElement(n) / (int)opnd2.getElement(n);
            pDst[n] = value;
        }
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }
}

void ETensorCore::m_add(ETensor opnd1, ETensor opnd2) {
    int64 ndat = opnd1.shape().valid_size();

    if (m_type == VDataType::float32) {
        float* pDst = m_data.float_ptr();

        for (int64 n = 0; n < ndat; n++) {
            float value = (float)opnd1.getElement(n) + (float)opnd2.getElement(n);
            pDst[n] = value;
        }
    }
    else if (m_type == VDataType::int64) {
        int64* pDst = m_data.int64_ptr();

        for (int64 n = 0; n < ndat; n++) {
            int64 value = (int64)opnd1.getElement(n) + (int64)opnd2.getElement(n);
            pDst[n] = value;
        }
    }
    else if (m_type == VDataType::int32) {
        int* pDst = m_data.int_ptr();

        for (int64 n = 0; n < ndat; n++) {
            int value = (int)opnd1.getElement(n) + (int)opnd2.getElement(n);
            pDst[n] = value;
        }
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }
}

void ETensorCore::m_subtract(ETensor opnd1, ETensor opnd2) {
    int64 ndat = opnd1.shape().valid_size();

    if (m_type == VDataType::float32) {
        float* pDst = m_data.float_ptr();

        for (int64 n = 0; n < ndat; n++) {
            float value = (float)opnd1.getElement(n) - (float)opnd2.getElement(n);
            pDst[n] = value;
        }
    }
    else if (m_type == VDataType::int64) {
        int64* pDst = m_data.int64_ptr();

        for (int64 n = 0; n < ndat; n++) {
            int64 value = (int64)opnd1.getElement(n) - (int64)opnd2.getElement(n);
            pDst[n] = value;
        }
    }
    else if (m_type == VDataType::int32) {
        int* pDst = m_data.int_ptr();

        for (int64 n = 0; n < ndat; n++) {
            int value = (int)opnd1.getElement(n) - (int)opnd2.getElement(n);
            pDst[n] = value;
        }
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }
}

void ETensorCore::m_max(ETensor opnd1, ETensor opnd2) {
    int64 ndat = opnd1.shape().valid_size();

    if (m_type == VDataType::float32) {
        float* pDst = m_data.float_ptr();
        float* pSrc1 = opnd1.float_ptr();
        float* pSrc2 = opnd2.float_ptr();

        float sum = 0;

        for (int64 n = 0; n < ndat; n++) {
            pDst[n] = MAX(pSrc1[n], pSrc2[n]);
        }
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }
}

