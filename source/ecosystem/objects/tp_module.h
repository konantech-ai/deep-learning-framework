#pragma once

#include "../utils/tp_common.h"
#include "../connect/tp_api_conn.h"

class EModuleCore;

class TpStreamOut;

class EModule {

public:
    EModule();
    EModule(ENN nn);
    EModule(ENN nn, VHModule hModule);
    EModule(const EModule& src);
    EModule(EModuleCore* core);
    virtual ~EModule();
    EModule& operator =(const EModule& src);
    operator VHModule();
    bool isValid();
    void close();
    ENN nn();
    EModuleCore* getCore();
    EModuleCore* cloneCore();
    int meNth();
    int meRefCnt();
    int handleNth();
    int handleRefCnt();
    EModuleCore* createApiClone();

protected:
    EModuleCore* m_core;

public:
    EModule(ENN nn, string sName, VDict kwArgs = {}); // not exported
    //EModule(string sName, string sBuiltin, VDict kwArgs = {});

    void setup(string sName, string sBuiltin, EModuleType moduleType, VDict kwArgs); // not exported

    string getName();

    EModuleType getType();

    EModule expand(VShape shape = {}, VDict kwArgs = {});
    //EModule expandMacro(VShape shape = {}, VDict kwArgs = {});
    EModule to(string device);

    VShape getInShape();
    VShape getOutShape();

    string __str__();
    string desc(int depth, int nth, int64& total_pm, string pos);
    string desc(string* p_name, string* p_builtin, VDict* p_option, VShape* p_in_shape, VShape* p_out_shape, int64* p_pmsize);
    
    void saveModel(TpStreamOut& fout);

    ETensor __call__(ETensor x);
    ETensorDict __call__(ETensorDict xs);

    //void backward();

    virtual ETensor forward(ETensor x);           // not exported, 파생 클래스의 forward 함수가 불리워야 하므로 virtual 한정자 생략 금지
    virtual ETensorDict forward(ETensorDict xs);   // not exported, 파생 클래스의 forward 함수가 불리워야 하므로 virtual 한정자 생략 금지

    void eval();     // 비학습모드로 전환
    void train(); // EOptimizer optimizer);    // 학습모드로 전환 (디폴트)

    void appendChild(EModule child);
    //void setName(string name); // 생성시 이름 지정 가능하며 그렇지 않은 경우 서버에서 이름이 자동 부여됨, 추후 불일치 방지 위해 변경 불허 

    ETensor predict(ETensor x);
    ETensorDict predict(ETensorDict xs);

    EParameters parameters();

    //VDict state_dict(); // 미구현 상태이며 구현 필요성 미흡

    EModule createClone();

    //EModule fetchChild(string name); // seekLayer()와 동일
    EModule fetch_child(string name);
    EModule seekLayer(string name);
    EModule nthChild(int nth);

    void init_parameters();

    //bool loadParameters(string filePath, string mode = ""); // yolo4.wiights의 경우처럼 타 기관에서 기학습한 파라미터 내용을 비교실험용으로 적재
    void load_cfg_config(string cfg_path, string weight_path);

    void saveParameters(string filePath);
    void saveParameters(string root, string filePath);

    int addForwardCallbackHandler(TCbForwardCallback* pCbFunc, VDict instInfo, VDict filters); // not ex[prted
    int addBackwardCallbackHandler(TCbBackwardCallback* pCbFunc, VDict instInfo, VDict filters); // not ex[prted

    void removeCallbackHandler(int nId); // not ex[prted

    void upload_data_index(VList dataIdx);

    void setParamater(ETensorDict paramTensors, string mode);

protected:
    //void m_fitTrain(int64 epoch, DataLoader dataloader, time_t startTime, VDict kwArgs);
    //void m_fitValidate(DataLoader dataloader, time_t startTime, VDict kwArgs);

    static mutex ms_moduleMutex;

    friend class ModuleExt;

};
