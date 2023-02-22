#pragma once

#include "../api/vcommon.h"
#include "../api_objects/vsession.h"
#include "../api_objects/vtensor.h"
#include "../local_objects/vgraph.h"
#include "../local_objects/vdevicemanager.h"
#include "../local_objects/vgraph_node.h"

class VGraphNodeCore : public VObjCore {
protected:
    friend class VGraphNode;
protected:
    VGraphNodeCore(VSession session, string sBuiltin = "", VDict kwArgs = {});
    VGraphNodeCore* clone() { return (VGraphNodeCore*)clone_core(); }
    VSession session() { return m_session; }
    void m_setup();
protected:
    VSession m_session;
    string m_sBuiltin;
    VDict m_propDict;

protected:
    VGraphOpCode m_opCode;

    VDeviceManager m_deviceMan;
    VGraphCore* m_pGraphCore;   // 객체 포인터가 아닌 객체 자체를 멤버로 할 경우 순환 참조가 생겨 메모리 반납에 실패함

    string m_varName;

    VTensor m_pm;	// pm 연산자의 경우에 한해 파라미터 텐서 지시, 사용 디바이스 메모리에 복사본 확보
    VTensor m_grad;	// pm 연산자의 경우에 한해 0번 디바이스의 공유 경사도 텐서 지시, 사용시 락킹 필수

    VDict m_nesterovPmSet; // nesterov 옵티마이저의 경우 velocity 정보 이용한 전처리 위해 보존
    VDict m_pmSet; // rnn 계열 레이어에서 층위별, 양방향 여부에 따라 달리 구성되는 파라미터들을 일괄 전달하기 위해서도 사용

    VValue m_aux;	// shape, list 연산자 등의 경우 필요한 타입으로 변환해 사용

    vector<VGraphNode> m_children;

    void m_setup(string sExp, VList params, int& pos);
    void m_setup(VGraphNodeCore* src_core);
    void m_seekOperands(string layerExp, VList params, int& pos);

    VDict m_pmsetToCurDevice(VCbBackInfo cbInfo, VExecTracer tracer);

    VGraphNode m_split_multi_operands(VGraphNodeCore* pSrc, VGraphOpCode opCode, vector<VGraphNode> children);

    static int ms_skipSpaces(string layerExp, int& pos);

};
