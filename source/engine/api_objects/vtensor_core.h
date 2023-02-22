#pragma once

#include "../api/vcommon.h"
#include "../api_objects/vsession.h"
#include "../local_objects/vtensordata.h"
#include "../api_objects/vmodule.h"
#include "../local_objects/vcbbackinfo.h"

class VTensorCore : public VObjCore {
protected:
    friend class VTensor;
protected:
    VTensorCore(VSession session, string sBuiltin = "", VDict kwArgs = {});
    VTensorCore* clone() { return (VTensorCore*)clone_core(); }
    ~VTensorCore();
    void m_onCreate();
    void m_onDelete();
    VSession session() { return m_session; }
protected:
    VSession m_session;
    string m_sBuiltin;
    VDict m_propDict;
    int m_nCheckCode;
    static int ms_nCheckCode;

protected:
    VTensorCore();

    VShape m_shape;
    VDataType m_dataType;

    VTensorData m_data;

    bool m_bNeedGrad;
    VGraphOpCode m_opCode;

    VTensorList m_operands;
    //VList m_children;

    // 모듈 캡슐 객체를 멤버로 가질 경우 순환참조로 인해 메모리 릭이 발생한다.
    VModuleCore* m_pOwningModule;

    //memo("초기화 관련 정보를 담아 쉽게 레셋 가능하게 하가.");

    VValue m_aux; // 연산 과정에 필요한 shape, list 등의 보조 정보를 빈 텐서에 이들 정보를 담아 활용한다.
    VDict m_opArgs; // 연산 과정에 필요한 shape, list 등의 보조 정보를 모아 첫번째 계산 대상 텐서(operands[0])에 담아 활용한다.
    VDict m_myArgs; // 연산 과정에 필요한 shape, list 등의 보조 정보를 모아 계산 결과 텐서(result)에 담아 활용한다.
    // 연산 준비 중에 수집되는 보조 정보는 result가 부재중이므로 이미 준비된 오퍼랜드의 m_args에 담아 전달한다.
    // 하지만 이 정보 저장 방식은 두 개 이상의 텐서 계산에 반복 활용되는 오퍼랜드 텐서의 경우 문제를 발생시킨다.
    // 따라서 계산 후에는 result에 담지만 result 역시 다른 연산의 피연산자가 될 수 있으므로 변수를 구별해 설치한다.
    VDict m_initArgs; // 초기화 방법에 대한 정보를 저장해 처리 중에 텐서 내용을 다시 초기화시켜야 하는 경우에 활용한다.

    int m_nOperandRefCount;

    vector<VCbBackSlot> m_cbBackSlots;

    static bool ms_bNoDump;
    static mutex ms_grad_mutex;

    void m_initParam(TensorInitMethod init_op, float mean, float init_arg, bool adaptive);
    void m_copyDataFrom(VTensorCore* src, VExecTracer tracer);
    void m_resetCbBackSlot(int sid);
};
