#pragma once

#include "../api/vcommon.h"
#include "../api_objects/vsession.h"

class VTensorDataCore : public VObjCore {
protected:
    friend class VTensorData;
protected:
    VTensorDataCore(VSession session, string sBuiltin = "", VDict kwArgs = {});
    VTensorDataCore* clone() { return (VTensorDataCore*)clone_core(); }
    VSession session() { return m_session; }
    void m_setup();
protected:
    VSession m_session;
    string m_sBuiltin;
    VDict m_propDict;

    int m_nDevice;

    void* m_ptr;
    int64 m_byteSize;

    virtual ~VTensorDataCore();

    void m_freeData();

    void m_memset(int value);
    void m_fill_float(float value);
    void m_init_random_normal(float mean, float init_arg, bool adaptive);
    void m_init_random_uniform(float mean, float init_arg);

    //void m_initParam(VTensorInit init_op, float mean, float init_arg, bool adaptive);
    //void m_uploadData(void* pData);
    //void m_downloadData(void* pData);
    //void m_setZero();
    //void m_init_random_normal(float mean, float std, bool adapt);
};
