#pragma once

#include "../utils/tp_common.h"
#include "../objects/tp_module.h"
#include "../connect/tp_nn_server.h"

class EFunction;
class ApiConn;

class ENNCore : public EcoObjCore {
protected:
    friend class ENN;
protected:
    ENNCore();
    ~ENNCore();
    ENNCore* clone() { return (ENNCore*)clone_core(); }
    void m_setup();
    void m_delete();
protected:
    string m_sEngineVersion;

    int m_nDeviceCount;

    ApiConn* m_pApiConn;

    VDict m_builtinNames;

protected:
    //static map<VHSession, VHandle> ms_sessionToNNMap;

    map<VHModule, EModule*> m_customModuleMap;   // value 부분은 EModule 클래스의 파생클래스 객체의 주소일 가능성이 크다
    map<VHFunction, EFunction*> m_userDefFuncMap;

    void m_customModuleExecCbFunc(ENN nn, void* pAux, time_t time, VHandle hModule, const VExBuf* pXsBuf, const VExBuf** ppYsBuf);
    void m_freeReportBufferCbFunc(ENN nn, void* pAux, const VExBuf* pResultBuf);
};
