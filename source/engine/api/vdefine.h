#pragma once

#ifdef USE_AGAIN
#include "../api/vcommon.h"
#include "../api_objects/vsession.h"

// 참조계수 오작동 디버깅을 위해 아래의 선언을 제거해 봄
//V##cls(VH##cls h##cls); \ // 오류유발자로 바꿈
//V##cls(V##cls##Core* core); \ // VObjCore* 이용으로 대치

#define DECLARE_API_CAPSULE(cls) \
class V##cls##Core; \
class V##cls { \
public: \
    V##cls(); \
    V##cls(VSession session, string sBuiltin, VDict kwArgs = {}); \
    V##cls(const V##cls& src); \
    V##cls(VSession session, VH##cls handle); \
    V##cls(V##cls##Core* core); \
    virtual ~V##cls(); \
    V##cls& operator =(const V##cls& src); \
    VH##cls cloneCore(); \
    VH##cls cloneHandle(); \
    V##cls##Core* getClone(); \
    V##cls##Core* getCore(); \
    bool isValid(); \
    void closeHandle(); \
    VSession session(); \
    int getNth(); \
    int getRefCnt(); \
    void incRefCount(); \
protected: \
    V##cls##Core* m_core; \
public:



#define DECLARE_API_CORE(cls) \
class V##cls##Core : public VObjCore { \
protected: \
    friend class V##cls; \
protected: \
    V##cls##Core(VSession session, string sBuiltin="", VDict kwArgs={}); \
    V##cls##Core* clone() { return (V##cls##Core*)clone_core(); } \
    ~V##cls##Core(); \
    void m_onCreate(); \
    void m_onDelete(); \
    VSession session() { return m_session; } \
protected: \
    VSession m_session; \
    string m_sBuiltin; \
    VDict m_propDict; \
    int m_nCheckCode; \
    static int ms_nCheckCode;

#define DECLARE_END() \
};


// 참조계수 오작동 디버깅을 위해 아래의 정의를 제거해 봄
//V##cls::V##cls(VH##cls h##cls) { \
//m_core = ((V##cls##Core*)h##cls)->clone(); } \
// V##cls::V##cls(V##cls##Core* core) { m_core = core->clone(); } \

#define DEFINE_API_OBJECT(cls) \
V##cls::V##cls() { m_core = NULL; } \
V##cls::V##cls(VSession session, string sBuiltin, VDict kwArgs) { \
    m_core = new V##cls##Core(session, sBuiltin, kwArgs); } \
V##cls::V##cls(const V##cls& src) { m_core = src.m_core->clone(); } \
V##cls::V##cls(VSession session, VH##cls handle) { \
    m_core = NULL; \
    V##cls##Core* core = (V##cls##Core*)handle; \
    if (core == NULL) VP_THROW1(VERR_INVALID_CORE, #cls); \
    if (core->m_nCheckCode != V##cls##Core::ms_nCheckCode) VP_THROW1(VERR_NOT_EQUAL_CORE_CHECKCODE, #cls); \
    if (core->m_session != session) VP_THROW1(VERR_NOT_EQUAL_CORE_SESSION, #cls); \
    m_core = (V##cls##Core*)core->clone_core(); } \
V##cls::V##cls(V##cls##Core* core) { m_core = core->clone(); } \
V##cls::~V##cls() { m_core->destroy(); } \
V##cls& V##cls::operator =(const V##cls& src) { \
    if (&src != this && m_core != src.m_core) { \
        m_core->destroy(); \
        m_core = src.m_core->clone(); } \
    return *this; } \
VH##cls V##cls::cloneCore() { \
    return (VH##cls)m_core->clone(); } \
VH##cls V##cls::cloneHandle() { \
    return (VH##cls)m_core->clone_handle(); } \
V##cls##Core* V##cls::getClone() { return (V##cls##Core*)m_core->clone_core(); } \
V##cls##Core* V##cls::getCore() { return m_core; } \
bool V##cls::isValid() { return m_core != NULL; } \
void V##cls::closeHandle() { if (this) { m_core->destroy_handle(); } } \
VSession V##cls::session() { return m_core->m_session; } \
int V##cls::getRefCnt() { return m_core->getRefCnt(); } \
int V##cls::getNth() { return m_core->getNth(); } \
void V##cls::incRefCount() { m_core->incRefCnt(); } \
V##cls##Core::V##cls##Core(VSession session, string sBuiltin, VDict kwArgs) : VObjCore(VObjType::cls) { \
    m_nCheckCode = ms_nCheckCode; \
    m_session = session; \
    m_sBuiltin = vutils.tolower(sBuiltin); \
    m_propDict = kwArgs; \
    m_onCreate(); } \
V##cls##Core::~V##cls##Core() { \
    m_onDelete(); \
    m_nCheckCode = 0;\
} \

#define DECLARE_HIDDEN_CAPSULE(cls) \
class V##cls##Core; \
class V##cls { \
public: \
    V##cls(); \
    V##cls(VSession session, string sBuiltin, VDict kwArgs = {}); \
    V##cls(const V##cls& src); \
    V##cls(V##cls##Core* core); \
    virtual ~V##cls(); \
    V##cls& operator =(const V##cls& src); \
    V##cls##Core* getClone(); \
    V##cls##Core* getCore(); \
    void destroyCore(); \
    VSession session() const; \
    bool isValid(); \
    int getRefCnt(); \
    int getNth(); \
protected: \
    V##cls##Core* m_core; \
public:

#define DECLARE_HIDDEN_CORE(cls) \
class V##cls##Core : public VObjCore { \
protected: \
    friend class V##cls; \
protected: \
    V##cls##Core(VSession session, string sBuiltin="", VDict kwArgs={}); \
    V##cls##Core* clone() { return (V##cls##Core*)clone_core(); } \
    VSession session() { return m_session; } \
    void m_setup(); \
protected: \
    VSession m_session; \
    string m_sBuiltin; \
    VDict m_propDict;

#define }; \
};

#define DEFINE_HIDDEN_OBJECT(cls) \
V##cls::V##cls() { m_core = NULL; } \
V##cls::V##cls(VSession session, string sBuiltin, VDict kwArgs) { \
    m_core = new V##cls##Core(session, sBuiltin, kwArgs); } \
V##cls::V##cls(const V##cls& src) { m_core = src.m_core->clone(); } \
V##cls::V##cls(V##cls##Core* core) { m_core = core->clone(); } \
V##cls::~V##cls() { m_core->destroy(); } \
V##cls& V##cls::operator =(const V##cls& src) { \
    if (&src != this && m_core != src.m_core) { \
        m_core->destroy(); \
        m_core = src.m_core->clone(); } \
    return *this; } \
V##cls##Core* V##cls::getClone() { return (V##cls##Core*)m_core->clone_core(); } \
V##cls##Core* V##cls::getCore() { return m_core; } \
void V##cls::destroyCore() { \
    if (m_core->getRefCnt() > 1) m_core->destroy(); \
    else { \
        m_core->destroy(); \
        m_core = NULL; \
    } \
} \
VSession V##cls::session() const { return m_core->m_session; } \
bool V##cls::isValid() { return m_core != NULL; } \
int V##cls::getRefCnt() { return m_core->getRefCnt(); } \
int V##cls::getNth() { return m_core->getNth(); } \
V##cls##Core::V##cls##Core(VSession session, string sBuiltin, VDict kwArgs) : VObjCore(VObjType::cls) { \
    m_sBuiltin = vutils.tolower(sBuiltin); \
    m_propDict = kwArgs; \
    m_session = session, \
    m_setup(); }


#define DECLARE_MISC_CAPSULE(cls) \
class V##cls##Core; \
class V##cls { \
public: \
    V##cls(); \
    V##cls(string sBuiltin, VDict kwArgs = {}); \
    V##cls(const V##cls& src); \
    V##cls(V##cls##Core* core); \
    virtual ~V##cls(); \
    V##cls& operator =(const V##cls& src); \
    V##cls##Core* getClone(); \
    bool isValid(); \
    int getRefCnt(); \
    int getNth(); \
protected: \
    V##cls##Core* m_core; \
public:

#define DECLARE_MISC_CORE(cls) \
class V##cls##Core : public VObjCore { \
protected: \
    friend class V##cls; \
protected: \
    V##cls##Core(string sBuiltin="", VDict kwArgs={}); \
    V##cls##Core* clone() { return (V##cls##Core*)clone_core(); } \
    void m_setup(); \
protected: \
    string m_sBuiltin; \
    VDict m_propDict;

#define }; \
};

#define DEFINE_MISC_OBJECT(cls) \
V##cls::V##cls() { m_core = NULL; } \
V##cls::V##cls(string sBuiltin, VDict kwArgs) { \
    m_core = new V##cls##Core(sBuiltin, kwArgs); } \
V##cls::V##cls(const V##cls& src) { m_core = src.m_core->clone(); } \
V##cls::V##cls(V##cls##Core* core) { m_core = core->clone(); } \
V##cls::~V##cls() { m_core->destroy(); } \
V##cls& V##cls::operator =(const V##cls& src) { \
    if (&src != this && m_core != src.m_core) { \
        m_core->destroy(); \
        m_core = src.m_core->clone(); } \
    return *this; } \
V##cls##Core* V##cls::getClone() { return (V##cls##Core*)m_core->clone_core(); } \
bool V##cls::isValid() { return m_core != NULL; } \
int V##cls::getRefCnt() { return m_core->getRefCnt(); } \
int V##cls::getNth() { return m_core->getNth(); } \
V##cls##Core::V##cls##Core(string sBuiltin, VDict kwArgs) : VObjCore(VObjType::cls) { \
    m_sBuiltin = vutils.tolower(sBuiltin); \
    m_propDict = kwArgs; \
    m_setup(); }
#endif
