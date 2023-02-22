#pragma once

#include "../utils/tp_common.h"

class TpStreamIn;
class TpStreamOut;

class ETensorCore;
class ETensor {
public:
    ETensor();
    ETensor(ENN nn);
    ETensor(ENN nn, VHTensor hTensor);
    ETensor(const ETensor& src);
    ETensor(ETensorCore* core);
    virtual ~ETensor();
    ETensor& operator =(const ETensor& src);
    operator VHTensor();
    bool isValid();
    void close();
    ENN nn();
    ETensorCore* getCore();
    ETensorCore* cloneCore();
    int meNth();
    int meRefCnt();
    int handleNth();
    int handleRefCnt();
    ETensorCore* createApiClone();

protected:
    ETensorCore* m_core;

public:
    //ETensor(VShape shape, VDataType type);
    ETensor(ENN nn, VShape shape, VDataType type, void* pData = NULL);
    ETensor(ENN nn, VShape shape, VDataType type, int device);
    ETensor(ENN nn, VShape shape, VDataType type, string initMethod);
    ETensor(ENN nn, VShape shape, VDataType dataType, VDataType loadType, FILE* fin);
    ETensor(ENN nn, VShape shape, VDataType type, VList values);
    ETensor(ETensor src, VShape shape);
    ETensor(ENN nn, VHTensor hTensor, bool needToClose, bool needToUpload);

    VShape shape() const;
    VDataType type() const;
    string type_desc() const;
    int64 byteSize();
    int device();
    string desc();
    int64 len();
    int64 size();

    bool hasNoData();
    void allocData();
    void downloadData();
    void downloadData(ENN nn);
    void copyData(VShape shape, VDataType type, void* pData);

    EScalar item();

    void* void_ptr();
    int* int_ptr();
    int64* int64_ptr();
    float* float_ptr();
    char* char_ptr();
    unsigned char* uchar_ptr();
    unsigned char* bool_ptr();

    virtual void backward();
    virtual void backwardWithGradient(ETensor grad);

    void upload(bool force=false);
    ETensor toDevice(int device);
    void dump(string title, bool full = false);
    string get_dump_str(string title, bool full = false);
    void free_dump_str(const char* pData);
    void dump_arr_feat(int nth, string title);

    void save(TpStreamOut* fout);

    static ETensor load(ENN nn, TpStreamIn* fin);
    static ETensor load(ENN nn, FILE* fin);

    VIntList findElement(VValue element);

    ETensor operator[](int64 index);
    ETensor operator[](ETensor index);

    //ETensorCore* detachCore();

    void fetchIdxRows(ETensor src, int64* pnMap, int64 size);
    void fetchIdxRows(ETensor src, int* pnMap, int64 size);
    
    void copy_into_row(int64 nthRow, ETensor src);

    void reset();

    operator int() const;
    operator float() const;
    
    int64 to_int64() const;

    ETensor to_type(VDataType type, string option);
    ETensor argmax(int64 axis);
    ETensor sum();
    ETensor mean();
    ETensor abs();
    ETensor sigmoid();
    ETensor square();
    ETensor max(ETensor rhs);
    ETensor transpose();
    ETensor transpose(VList axes);

    ETensor set_type(VDataType type, string option);
    ETensor reshape(VShape shape);

    ETensor operator ==(ETensor rhs);
    ETensor operator ==(int value);
    ETensor operator ==(int64 value);
    ETensor operator ==(float value);

    ETensor operator !=(ETensor rhs);
    ETensor operator !=(int value);
    ETensor operator !=(int64 value);
    ETensor operator !=(float value);

    ETensor operator >(ETensor rhs);
    ETensor operator >(int value);
    ETensor operator >(int64 value);
    ETensor operator >(float value);

    ETensor operator >=(ETensor rhs);
    ETensor operator >=(int value);
    ETensor operator >=(int64 value);
    ETensor operator >=(float value);

    ETensor operator <(ETensor rhs);
    ETensor operator <(int value);
    ETensor operator <(int64 value);
    ETensor operator <(float value);

    ETensor operator <=(ETensor rhs);
    ETensor operator <=(int value);
    ETensor operator <=(int64 value);
    ETensor operator <=(float value);

    ETensor operator &&(ETensor rhs);
    ETensor operator ||(ETensor rhs);
    ETensor operator &(ETensor rhs);
    ETensor operator |(ETensor rhs);

    ETensor operator*(ETensor rhs);
    ETensor operator*(int rhs);
    ETensor operator*(int64 rhs);
    ETensor operator*(float rhs);

    ETensor operator/(ETensor rhs);
    ETensor operator/(int rhs);
    ETensor operator/(int64 rhs);
    ETensor operator/(float rhs);

    ETensor operator+(ETensor rhs);
    ETensor operator+(float rhs);
    ETensor operator+(int rhs);
    ETensor operator+(int64 rhs);

    ETensor operator-(ETensor rhs);
    ETensor operator-(int rhs);
    ETensor operator-(int64 rhs);
    ETensor operator-(float rhs);

    friend ETensor operator*(int lhs, ETensor rhs);

    //friend ETensor operator*(ETensor lhs, ETensor const& rhs);
    friend ETensor operator*(float lhs, ETensor const& rhs);
    friend ETensor operator*(double lhs, ETensor const& rhs);
    //friend TTensor& operator*(TParam lhs, TTensor const& rhs);
    friend ETensor operator^(ETensor lhs, int rhs);
    //friend ETensor operator ==(ETensor lhs, ETensor rhs);

    friend ETensor abs(ETensor term);
    friend ETensor max(ETensor lhs, ETensor rhs);

    void setElement(int64 pos, VValue value);
    void setElement(VList pos, VValue value);
    void setSlice(VList sliceIndex, VDataType type, int64 datSize, void* value_ptr);

    VValue getElement(int64 pos);
    VValue getElement(VList pos);
    ETensor getSlice(VList sliceIndex);

    void setDebugName(string name);

    void copyPartialData(ETensor src);
    void copyRowFrom(int64 destRow, ETensor src, int64 srcRows);
    
    void shift_timestep_to_right(ETensor src, int64 steps);

    void shuffle();

    static ETensor ones(ENN nn, VShape shape);
    static ETensor zeros(ENN nn, VShape shape);
    static ETensor rand_uniform(ENN nn, VShape shape, float min = 0.0f, float max = 1.0f);
    static ETensor rand_normal(ENN nn, VShape shape, float avg, float std);
    static ETensor rand_onehot(ENN nn, VShape shape);
    static ETensor arange(ENN nn, VValue arg1, VValue arg2 = VValue(), VValue arg3 = VValue(), VDict kwArgs = {});
    static ETensor matmul(ETensor x1, ETensor x2);
    static ETensor hconcat(ETensor x1, ETensor x2);
    static ETensor vstack(ETensor x1, ETensor x2);
    static ETensor tile(ETensor x, int64 rep);
    static ETensor repeat(ETensor x, int64 rep);
    static ETensor resize(ETensor x, VShape shape);

    void vstack_on(ETensor x1, ETensor x2);
    void rand_uniform_on(float min = 0.0f, float max = 1.0f);

    ETensor sin();

    static ETensor evaluate(string expression, VList args);
    static ETensor linspace(float from, float to, int count, VDict kwArgs);
    static ETensor randh(VShape shape, VDict kwArgs);

protected:
    void m_dumpElement(void* pHostBuf, int64 n);
    string m_getElementDesc(void* pHostBuf, int64 n);
    VDataType m_op_result_type(VDataType type1, VDataType type2);
    void m_dump(VShape shape, void* pHostBuf, int64 nth, int indent, bool full);
    string m_dump_to_str(VShape shape, void* pHostBuf, int64 nth, int indent, bool full);  // 파이썬 print() 지원을 위한 추가 구현
    void m_dumpNthElement(void* pHostBuf, int64 nth, bool bSpaceOnLeft = true);
    string m_dumpNthElement_to_str(void* pHostBuf, int64 nth, bool bSpaceOnLeft = true);
    VShape m_getSliceShape(VList sliceIndex, VShape srcShape);
};
