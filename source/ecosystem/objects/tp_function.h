#pragma once

#include "../utils/tp_common.h"

class EFunctionCore; \
    class EFunction {
    \
    public: \
        EFunction(); \
        EFunction(ENN nn); \
        EFunction(ENN nn, VHFunction hFunction); \
        EFunction(const EFunction& src); \
        EFunction(EFunctionCore* core); \
        virtual ~EFunction(); \
        EFunction& operator =(const EFunction& src); \
        operator VHFunction(); \
        bool isValid(); \
        void close(); \
        ENN nn(); \
        EFunctionCore* getCore(); \
        EFunctionCore* cloneCore(); \
        int meNth(); \
        int meRefCnt(); \
        int handleNth(); \
        int handleRefCnt(); \
        EFunctionCore* createApiClone(); \
    protected: \
        EFunctionCore* m_core; \
    public: \

    public:
        EFunction(ENN nn, string sBuiltin, string sName, VDict kwArgs);

        string getInstName();

        void registUserDefFunc(EFunction* pInst);

        virtual ETensor forward(int nInst, ETensor x, VDict opArgs);
        virtual ETensor forward(int nInst, ETensorList operands, VDict opArgs);

        virtual ETensor backward(int nInst, ETensor ygrad, ETensor x, VDict opArgs);
        virtual ETensor backward(int nInst, ETensor ygrad, int nth, ETensorList operands, VDict opArgs);

};
