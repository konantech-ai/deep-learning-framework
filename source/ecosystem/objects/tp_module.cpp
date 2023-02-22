#include "../objects/tp_module.h"
#include "../objects/tp_module_core.h"
#include "../objects/tp_tensor.h"
//#include "../objects/tp_parameters.h"
#include "../objects/tp_nn.h"
//#include "../objects/tp_dataloader.h"
//#include "../objects/tp_batchdata.h"
#include "../objects/tp_scalar.h"
#include "../connect/tp_api_conn.h"
#include "../utils/tp_stream.h"
#include "../utils/tp_utils.h"
#include "../utils/tp_exception.h"

EModule::EModule() { m_core = NULL; }
EModule::EModule(ENN nn) { m_core = new EModuleCore(nn, 0); }
EModule::EModule(ENN nn, VHModule hModule) { m_core = new EModuleCore(nn, hModule); }
EModule::EModule(const EModule& src) { m_core = src.m_core->clone(); }
EModule::EModule(EModuleCore* core) { m_core = core->clone(); }

EModule::~EModule() { m_core->destroy(); }

EModule& EModule::operator =(const EModule& src) {
    if (&src != this && m_core != src.m_core) {

        m_core->destroy();
        m_core = src.m_core->clone();
    }
    return *this;
}

EModule::operator VHModule() { return (VHModule)m_core->m_hEngineHandle; }
bool EModule::isValid() { return m_core != NULL; }
void EModule::close() { if (this) m_core->destroy(); }
ENN EModule::nn() { return m_core ? m_core->m_nn : ENN(); }
EModuleCore* EModule::getCore() { return m_core; }
EModuleCore* EModule::cloneCore() { return (EModuleCore*)m_core->clone(); }
int EModule::meNth() { return m_core->getNth(); }
int EModule::meRefCnt() { return m_core->getRefCnt(); }

EModuleCore::EModuleCore(ENN nn, VHModule hEModule) : EcoObjCore(VObjType::custom) {
    m_nn = nn;
    m_hEngineHandle = hEModule;
    m_setup();
}
EModuleCore::~EModuleCore() {
    m_delete();
    m_nn.getApiConn()->Module_close(m_hEngineHandle, __FILE__, __LINE__);
}

EModuleCore* EModule::createApiClone() { return m_core->clone(); }

//-----------------------------------------------------------------------------------------------------
// Capsule part

mutex EModule::ms_moduleMutex;

EModule::EModule(ENN nn, string sName, VDict kwArgs) {
    string sname;

    VHModule hModule = nn.getApiConn()->Module_create("custom", &sName, kwArgs, __FILE__, __LINE__);

    m_core = new EModuleCore(nn, hModule);
    m_core->m_setup(sName, "custom", EModuleType::custom, kwArgs);

    nn.registModuleMap(hModule, this);
}

void EModule::setup(string sName, string sBuiltin, EModuleType moduleType, VDict kwArgs) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    m_core->m_setup(sName, sBuiltin, moduleType, kwArgs);
}

string EModule::getName() {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    return m_core->m_sName;
}

EModuleType EModule::getType() {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    return m_core->m_moduleType;
}

EModule EModule::expand(VShape shape, VDict kwArgs) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    VHModule hExpandedModule = nn().getApiConn()->Module_expand(m_core->m_hEngineHandle, shape, kwArgs, __FILE__, __LINE__);
    return EModule(nn(), hExpandedModule);
}

/*
EModule EModule::expandMacro(VShape shape, VDict kwArgs) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    VHModule hExpandedModule = nn().getApiConn()->Module_expandMacro(m_core->m_hEngineHandle, shape, kwArgs, __FILE__, __LINE__);
    return EModule(nn(), hExpandedModule);
}
*/

EModule EModule::to(string device) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    VHModule hDeviceModule = nn().getApiConn()->Module_toDevice(m_core->m_hEngineHandle, device, __FILE__, __LINE__);
    return EModule(nn(), hDeviceModule);
}

ETensor EModule::__call__(ETensor x) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    x.upload();
    VHTensor yHandle = nn().getApiConn()->Module_evaluate(m_core->m_hEngineHandle, m_core->m_train, x, __FILE__, __LINE__);
    
    //m(" 미니배치 처리 때마다 새로 할당되고 있음:  메모리 반납 확인 필요");
    //m("버퍼 하나 잡아놓고 재활용 가능하도록 해 본다.");
    //static ETensor y_buffer;
    static VHTensor lastHandle = 0;
    static ETensor lastTensor;

    if (yHandle != lastHandle) {
        ETensor y(nn(), yHandle, true, false);
        y.downloadData();

        lastHandle = yHandle;
        lastTensor = y;

        return y;
    }
    else {
        ETensor y = lastTensor;
        y.downloadData();
        return y;
    }
}

ETensorDict EModule::__call__(ETensorDict xs) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    VDict xHandles = TpUtils::TensorDictToDict(xs, true);
    VDict yHandles = nn().getApiConn()->Module_evaluateEx(m_core->m_hEngineHandle, m_core->m_train, xHandles, __FILE__, __LINE__);

    ETensorDict ys = TpUtils::DictToTensorDict(nn(), yHandles);

    return ys;
}


ETensor EModule::predict(ETensor x) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);

    EModule model = *this;

    model.eval();

    nn().set_no_grad();

    ETensor pred = model.__call__(x);

    return pred;
}

ETensorDict EModule::predict(ETensorDict xs) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);

    EModule model = *this;

    model.eval();

    nn().set_no_grad();

    ETensorDict preds = model.__call__(xs);

    return preds;
}

ETensor EModule::forward(ETensor x) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    if (m_core->m_moduleType == EModuleType::custom) {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }

    return __call__(x);
}

ETensorDict EModule::forward(ETensorDict xs) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    if (m_core->m_moduleType == EModuleType::custom) {
        // 커스텀 레이어의 경우 forward 함수가 오버르드 선언되어 있어야 한다.
        if (xs.size() > 1) TP_THROW(VERR_UNDEFINED);
        // ETensorDict 입출력의 forward 함수가 오버르드 선언되어 있지 않으므로
        // Tensort 입출력의 forward 함수가 오버로드 선언되어 있는 경우에 대비해 아래 호출
        ETensor x = xs.begin()->second;
        ETensor y = forward(x);

        return { {xs.begin()->first, y}};
    }

    for (auto& it: xs) it.second.upload();

    return __call__(xs);
}

void EModule::setParamater(ETensorDict paramTensors, string mode) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);

    for (auto& it : paramTensors) {
        //it.second.dump(it.first);
        it.second.upload();
    }

    VDict tHandles = TpUtils::TensorDictToDict(paramTensors, true);

    nn().getApiConn()->Module_setParamater(m_core->m_hEngineHandle, tHandles, mode, __FILE__, __LINE__);
}

string EModule::__str__() {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    int64 total_pm = 0;
    return m_core->m_desc(0, -1, total_pm, "#layer");
}

string EModule::desc(int depth, int nth, int64& total_pm, string pos) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    return m_core->m_desc(depth, nth, total_pm, pos);
}

string EModule::desc(string* p_name, string* p_builtin, VDict* p_option, VShape* p_in_shape, VShape* p_out_shape, int64* p_pmsize) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    TP_THROW(VERR_NOT_IMPLEMENTED_YET);
}

void EModule::saveModel(TpStreamOut& fout) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    return m_core->m_saveModel(fout);
}

VShape EModule::getInShape() {
    VDict dict;
    nn().getApiConn()->Module_getModuleInfo(m_core->m_hEngineHandle, dict, __FILE__, __LINE__);
    VShape shape = dict["inshape"];
    return shape.copy();
}

VShape EModule::getOutShape() {
    VDict dict;
    nn().getApiConn()->Module_getModuleInfo(m_core->m_hEngineHandle, dict, __FILE__, __LINE__);
    VShape shape = dict["outshape"];
    return shape.copy();
}

EModule EModule::createClone() {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    EModule clone = nn().createModule(m_core->m_sBuiltin, m_core->m_kwArgs);
    nn().getApiConn()->Module_copyChildren((VHModule)clone, m_core->m_hEngineHandle, __FILE__, __LINE__);
    
    /*
    for (auto& it: m_core->m_children) {
        EModule childClone = it.createClone();
        clone.addChild("", childClone);
    }
    */
    return clone;
}

EModule EModule::seekLayer(string sSeekName) {
    if (m_core == NULL) TP_THROW(VERR_UNDEFINED);

    VHModule hChildModule = nn().getApiConn()->Module_fetchChild(m_core->m_hEngineHandle, sSeekName, false, __FILE__, __LINE__);

    EModule child = EModule(nn(), hChildModule);
    child.m_core->m_sName = sSeekName;

    return child;
}

EModule EModule::fetch_child(string sSeekName) {
    if (m_core == NULL) TP_THROW(VERR_UNDEFINED);

    VHModule hChildModule = nn().getApiConn()->Module_fetchChild(m_core->m_hEngineHandle, sSeekName, true, __FILE__, __LINE__);

    EModule child = EModule(nn(), hChildModule);
    child.m_core->m_sName = sSeekName;

    return child;
}

EModule EModule::nthChild(int nth) {
    if (m_core == NULL) TP_THROW(VERR_UNDEFINED);
    
    VList children = nn().getApiConn()->Module_getChildrenModules(m_core->m_hEngineHandle, __FILE__, __LINE__);

    if (children.size() <= nth) TP_THROW(VERR_UNDEFINED);

    EModule child = EModule(nn(), (VHModule)children[nth]);

    return child;
}

/*
EModule EModule::nthChild(int nth) {
    if (m_core == NULL) TP_THROW(VERR_UNDEFINED);
    //name = TpUtils::tolower(name);

    if (nth >= 0 && nth < (int64)m_core->m_children.size()) {
        return EModule(nn(), (VHModule)m_core->m_children[nth]);
    }

    TP_THROW(VERR_UNDEFINED);

    VHModule hChildModule = nn().getApiConn()->Module_fetchChild(m_core->m_hEngineHandle, sSeekName, __FILE__, __LINE__);
    printf("hChildModule = 0x%llx", (int64)hChildModule);

    EModule child = EModule(nn(), hChildModule);
    child.m_core->m_sName = sSeekName;

    printf("child.name = %s", child.getName().c_str());

    return child;
}
*/

/*
EModule EModule::fetchChild(string name) {
    for (auto& it : m_core->m_children) {
        EModule module(it);
        if (module.getName() == name) return module;
    }

    TP_THROW(VERR_UNDEFINED);
}

EModule EModule::operator [](int64 index) {
    if (index >= 0 && index < (int64)m_core->m_children.size()) {
        return EModule(nn(), (VHModule)m_core->m_children[index]);
    }
    TP_THROW(VERR_UNDEFINED);
}
*/

// 생성시 이름 지정 가능하며 그렇지 않은 경우 서버에서 이름이 자동 부여됨, 추후 불일치 방지 위해 변경 불허 
/*
void EModule::setName(string name) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    m_core->m_sName = name;
}
*/

void EModule::eval() {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    m_core->m_train = false;
}

void EModule::train() {//EOptimizer optimizer) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    m_core->m_train = true;
    //m_core->m_optimizer = optimizer;
}

void EModule::appendChild(EModule child) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    //m_core->m_children.push_back(child);
    nn().getApiConn()->Module_appendChildModule(m_core->m_hEngineHandle, child.m_core->m_hEngineHandle, __FILE__, __LINE__);
}

EParameters EModule::parameters() {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    VHParameters hParameters = nn().getApiConn()->Module_getParameters(m_core->m_hEngineHandle, __FILE__, __LINE__);
    return EParameters(nn(), hParameters);
}

/*
VDict EModule::state_dict() {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    printf("EModuleCore::state_dict() function not implemented yet.");
    //TP_THROW(KERR_UNIMPLEMENTED_YET);
    return VDict();
}
*/

VList m_parseDarknetCfg(string cfg_path) {
    VList blocks;
    VDict block;
    size_t pos;

    VStrList lines = TpUtils::read_file_lines(cfg_path);

    for (auto& line : lines) {
        TpUtils::trim(line);

        switch (line[0]) {
        case '[':
            block = { {"type", line} };
            blocks.push_back(block);
            break;
        case '\0':
        case '#':
        case ';':
            break;
        default:
            if ((pos = line.find('=')) < 0) {
                TP_THROW(VERR_UNDEFINED);
            }
            block[line.substr(0, pos)] = line.substr(pos+1);
            break;
        }
    }

    return blocks;
}

void m_read_cfg_weight(ENN nn, VList params, int64 index_from, int64 count, ETensorDict tensors, string type, FILE* fp) {
    for (int64 np = index_from; np < index_from + count; np++) {
        VDict paramInfo = params[np];
        if ((string)paramInfo["type"] == type) {
            ETensor tensor = tensors[paramInfo["key"]];
            float* pt = tensor.float_ptr();
            int64 nsize = tensor.shape().total_size();
            if (type == "moving_stat") {
                TP_THROW(VERR_UNDEFINED);   // moving stat will be split into mavg and mvar
                float* pTemp = new float[nsize];
                if (fread(pTemp, sizeof(float), nsize, fp) != nsize) {
                    TP_THROW(VERR_UNDEFINED);
                }
                int64 nhalf = nsize / 2;
                for (int64 n = 0; n < nhalf; n++) {
                    pt[2 * n + 0] = pTemp[n];
                    pt[2 * n + 1] = pTemp[n + nhalf];
                }
                delete[] pTemp;
            }
            else if (type == "w") {
                if (fread(pt, sizeof(float), nsize, fp) != nsize) {
                    TP_THROW(VERR_UNDEFINED);
                }
            }
            else {
                if (fread(pt, sizeof(float), nsize, fp) != nsize) {
                    TP_THROW(VERR_UNDEFINED);
                }
            }
            return;
        }
    }
    TP_THROW(VERR_UNDEFINED);
}

void EModule::load_cfg_config(string cfg_path, string weight_path) {
    if (m_core == NULL) TP_THROW(VERR_UNDEFINED);

    if (!TpUtils::file_exist(cfg_path)) TP_THROW(VERR_UNDEFINED);
    if (!TpUtils::file_exist(weight_path)) TP_THROW(VERR_UNDEFINED);

    VList blocks = m_parseDarknetCfg(cfg_path);

    int64 n = blocks.size() -  1;

    ETensorDict tensors;
    VList params = parameters().weightList(tensors);

    FILE* fp = TpUtils::fopen(weight_path, "rb");

    if (fp == NULL) TP_THROW(VERR_UNDEFINED);

    int major;
    int minor;
    int revision;

    fread(&major, sizeof(int), 1, fp);
    fread(&minor, sizeof(int), 1, fp);
    fread(&revision, sizeof(int), 1, fp);

    if ((major * 10 + minor) >= 2 && major < 1000 && minor < 1000) {
        size_t iseen = 0;
        fread(&iseen, sizeof(size_t), 1, fp);
    }
    else {
        int iseen = 0;
        fread(&iseen, sizeof(int), 1, fp);
    }

    int transpose = (major > 1000) || (minor > 1000);

    int64 block_count = blocks.size();
    int64 param_count = params.size();
    int64 param_index = 0;

    ENN nn = this->nn();

    for (int64 nb = 0; nb < block_count; nb++) {
        VDict block_info = blocks[nb];
        string block_type = block_info["type"];

        if (block_type == "[convolutional]" || block_type == "[deconvolutional]") {
            /*
            for (int64 np = param_index; np < param_count; np++) {
                VDict paramInfo = params[np];
                ETensor tensor = tensors[paramInfo["key"]]; // (nn, (VHTensor)paramInfo["tensor"]);
                if (np > 10) break;
            }
            */
            if ((string)TpUtils::seekDict(block_info, "batch_normalize", "") == "1") {
                m_read_cfg_weight(nn, params, param_index, 6, tensors, "shift", fp);
                m_read_cfg_weight(nn, params, param_index, 6, tensors, "rescale", fp);
                m_read_cfg_weight(nn, params, param_index, 6, tensors, "mavg", fp);
                m_read_cfg_weight(nn, params, param_index, 6, tensors, "mvar", fp);
                m_read_cfg_weight(nn, params, param_index, 6, tensors, "w", fp);

                param_index += 6;
            }
            else {
                m_read_cfg_weight(nn, params, param_index, 2, tensors, "b", fp);
                m_read_cfg_weight(nn, params, param_index, 2, tensors, "w", fp);

                param_index += 2;
            }
        }
        else if (block_type == "[connected]") {
            TP_THROW(VERR_NOT_IMPLEMENTED_YET);
            //load_connected_weights(l, fp, transpose);
        }
        else if (block_type == "[batchnorm]") {
            TP_THROW(VERR_NOT_IMPLEMENTED_YET);
            //load_batchnorm_weights(l, fp);
        }
        else if (block_type == "[crnn]") {
            TP_THROW(VERR_NOT_IMPLEMENTED_YET);
            //load_convolutional_weights(*(l.input_layer), fp);
            //load_convolutional_weights(*(l.self_layer), fp);
            //load_convolutional_weights(*(l.output_layer), fp);
        }
        else if (block_type == "[rnn]") {
            TP_THROW(VERR_NOT_IMPLEMENTED_YET);
            //load_connected_weights(*(l.input_layer), fp, transpose);
            //load_connected_weights(*(l.self_layer), fp, transpose);
            //load_connected_weights(*(l.output_layer), fp, transpose);
        }
        else if (block_type == "[lstm]") {
            TP_THROW(VERR_NOT_IMPLEMENTED_YET);
            //load_connected_weights(*(l.wi), fp, transpose);
            //load_connected_weights(*(l.wf), fp, transpose);
            //load_connected_weights(*(l.wo), fp, transpose);
            //load_connected_weights(*(l.wg), fp, transpose);
            //load_connected_weights(*(l.ui), fp, transpose);
            //load_connected_weights(*(l.uf), fp, transpose);
            //load_connected_weights(*(l.uo), fp, transpose);
            //load_connected_weights(*(l.ug), fp, transpose);
        }
        else if (block_type == "[gru]") {
            TP_THROW(VERR_NOT_IMPLEMENTED_YET);
            //load_connected_weights(*(l.wz), fp, transpose);
            //load_connected_weights(*(l.wr), fp, transpose);
            //load_connected_weights(*(l.wh), fp, transpose);
            //load_connected_weights(*(l.uz), fp, transpose);
            //load_connected_weights(*(l.ur), fp, transpose);
            //load_connected_weights(*(l.uh), fp, transpose);
        }
        else if (block_type == "[local]") {
            TP_THROW(VERR_NOT_IMPLEMENTED_YET);
            //int locations = l.out_w * l.out_h;
            //int size = l.size * l.size * l.c * l.n * locations;
            //fread(l.biases, sizeof(float), l.outputs, fp);
            //fread(l.weights, sizeof(float), size, fp);
        }
        else if (block_type == "[maxpool]") {
            // do nothing
        }
        else if (block_type == "[upsample]") {
            // do nothing
        }
        else if (block_type == "[route]") {
            // do nothing
        }
        else if (block_type == "[shortcut]") {
            // do nothing
        }
        else if (block_type == "[net]") {
            // do nothing
        }
        else if (block_type == "[yolo]") {
            // do nothing
        }
        else {
            TP_THROW(VERR_NOT_IMPLEMENTED_YET);
        }
    }

    char temp;
    if (fread(&temp, 1, 1, fp) != 0) {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }

    fclose(fp);

    for (auto& it : tensors) {
        if (!it.second.hasNoData()) {
            it.second.upload(true);
        }
    }
}

void EModule::saveParameters(string filePath) {
    if (m_core == NULL) TP_THROW(VERR_UNDEFINED);

    string root = "";
    
    if (filePath[0] != '/' && filePath[0] != '\\' && filePath.length() > 1 && filePath[1] != ':') {
        root = string(getenv("KONANAI_PATH")) + "/work/models/";
    }

    saveParameters(root, filePath);
}

void EModule::saveParameters(string root, string fileName) {
    if (m_core == NULL) TP_THROW(VERR_UNDEFINED);

    TpUtils::mkdir(root);

    string ext = TpUtils::getFileExt(fileName);
    string filePath = root + fileName;

    if (ext == "kon") {
        EParameters params = parameters();
        ETensorDict paramsInServer = params.weightDict();

        for (auto& it : paramsInServer) {
            it.second.downloadData();
        }

        VDict dict;
        nn().getApiConn()->Module_getModuleInfo(m_core->m_hEngineHandle, dict, __FILE__, __LINE__);
        VShape expandShape = dict["expand_shape"];

        TpStreamOut fout(nn(), filePath);

        fout.save_string("KonanAI model");
        fout.save_int(57578947);
        fout.save_string(nn().get_engine_version());

        saveModel(fout);

        fout.save_shape(expandShape);
        fout.save_string("KonanAI param");
        fout.save_int(37162886);
        fout.save_string(nn().get_engine_version());

        fout.save_tensordict(paramsInServer);
    }
    else {
        TP_THROW(VERR_NOT_IMPLEMENTED_YET);
    }
}

void EModule::init_parameters() {
    try {
        if (m_core == NULL) TP_THROW(VERR_UNDEFINED);
        
        EParameters params = parameters();
        params.initWeights();
    }
    catch (...) {
        TP_THROW(VERR_UNDEFINED);
    }
}

int EModule::addForwardCallbackHandler(TCbForwardCallback* pCbFunc, VDict instInfo, VDict filters) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    return nn().getApiConn()->Module_addForwardCallbackHandler(m_core->m_hEngineHandle, pCbFunc, filters, instInfo, __FILE__, __LINE__);
}

int EModule::addBackwardCallbackHandler(TCbBackwardCallback* pCbFunc, VDict instInfo, VDict filters) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    return nn().getApiConn()->Module_addBackwardCallbackHandler(m_core->m_hEngineHandle, pCbFunc, filters, instInfo, __FILE__, __LINE__);
}

void EModule::removeCallbackHandler(int nId) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    nn().getApiConn()->Module_removeCallbackHandler(m_core->m_hEngineHandle, nId, __FILE__, __LINE__);
}

void EModule::upload_data_index(VList dataIdx) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);
    nn().getApiConn()->Module_uploadDataIndex(m_core->m_hEngineHandle, dataIdx, __FILE__, __LINE__);
}

/*
void EModule::compile(VDict kwArgs) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);

    for (auto& it : kwArgs) {
        if (it.first == "loss") {
            if (it.second == "mse") m_core->m_loss = nn().MSELoss();
            else if (it.second == "mse") m_core->m_loss = nn().MSELoss();
            else if (it.second == "crossentropy") m_core->m_loss = nn().CrossEntropyLoss();
            else if (it.second == "crossentropy_idx") m_core->m_loss = nn().CrossEntropyPositiveIdxLoss();
            else if (it.second == "crossentropy_sigmoid") m_core->m_loss = nn().CrossEntropySigmoidLoss();
            else m_core->m_loss = nn().CustomLoss(it.second);
        }
        else if (it.first == "optimizer") {
            if (it.second == "sgd") m_core->m_optimizer = EOptimizer::SGD(parameters(), {});
            else if (it.second == "adam") m_core->m_optimizer = nn.AdamOptimizer(parameters(), {});
            else if (it.second == "momentum") m_core->m_optimizer = EOptimizer::Momentum(parameters(), {});
            else if (it.second == "nesterov") m_core->m_optimizer = EOptimizer::Nesterov(parameters(), {});
            else {
                TP_THROW(VERR_NOT_IMPLEMENTED_YET);
            }
        }
    }
}
*/

/*
void EModule::fit_ext(DataLoader* pTrLoader, DataLoader* pTeLoader, VDict kwArgs) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);

    DataLoader trDataloader = *pTrLoader;
    DataLoader teDataloader = *pTeLoader;

    int64 batch_size = TpUtils::seekDict(kwArgs, "batch_size", 32);
    int64 epochs = TpUtils::seekDict(kwArgs, "epochs", 1);

    float validation_split = TpUtils::seekDict(kwArgs, "validation_split", 0); // .1f);

    float learning_rate = TpUtils::seekDict(kwArgs, "learning_rate", 0);
    learning_rate = TpUtils::seekDict(kwArgs, "lr", learning_rate);
    if (learning_rate > 0) m_core->m_optimizer.setOption({ {"lr", learning_rate} });

    if (!m_core->m_loss.isValid()) TP_THROW(VERR_INVALID_LOSS_CORE);
    if (!m_core->m_optimizer.isValid()) TP_THROW(VERR_INVALID_OPTIMIZER_CORE);

    time_t startTime = time(NULL);

    for (int64 n = 0; n < epochs; n++) {
        m_fitTrain(n, trDataloader, startTime, kwArgs);
        if (teDataloader.isValid()) {
            m_fitValidate(teDataloader, startTime, kwArgs);
        }
    }
}

void EModule::fit(VDict kwArgs) {
    if (m_core == NULL) TP_THROW(VERR_INVALID_CORE);

    EModule model = *this;

    ETensor x((ETensorCore*)(VObjCore*)kwArgs["x"]);
    ETensor y((ETensorCore*)(VObjCore*)kwArgs["y"]);

    int64 batch_size = TpUtils::seekDict(kwArgs, "batch_size", 32);
    int64 epochs = TpUtils::seekDict(kwArgs, "epochs", 1);

    float validation_split = TpUtils::seekDict(kwArgs, "validation_split", 0); // .1f);

    float learning_rate = TpUtils::seekDict(kwArgs, "learning_rate", 0);
    learning_rate = TpUtils::seekDict(kwArgs, "lr", learning_rate);
    if (learning_rate > 0) m_core->m_optimizer.setOption({ {"lr", learning_rate} });

    //epochs = 1,
    //verbose = "auto",
    //callbacks = None,
    //validation_split = 0.0,
    //validation_data = None,
    //shuffle = True,
    //class_weight = None,
    //sample_weight = None,
    //initial_epoch = 0,
    //steps_per_epoch = None,
    //validation_steps = None,
    //validation_batch_size = None,
    //validation_freq = 1,
    //max_queue_size = 10,
    //workers = 1,
    //use_multiprocessing = False,

    if (!m_core->m_loss.isValid()) TP_THROW(VERR_INVALID_LOSS_CORE);
    if (!m_core->m_optimizer.isValid()) TP_THROW(VERR_INVALID_OPTIMIZER_CORE);

    DataLoader trainDataloader(nn(), x, y, { {"batch_size", batch_size},  {"ratio", 1.0f - validation_split}});
    DataLoader validDataloader(trainDataloader, { {"batch_size", batch_size} });
    
    time_t startTime = time(NULL);

    for (int64 n = 0; n < epochs; n++) {
        m_fitTrain(n, trainDataloader, startTime, kwArgs);
        if (validDataloader.isValid()) {
            m_fitValidate(validDataloader, startTime, kwArgs);
        }
    }
}

void EModule::m_fitTrain(int64 epoch, DataLoader dataloader, time_t startTime, VDict kwArgs) {
    int64 data_count = dataloader.data_count();
    int64 batch_count = dataloader.batch_count();
    int64 current = 0;
    int64 batch = 0;
    int64 batch_report = TpUtils::seekDict(kwArgs, "batch_report", 0);
    int64 epoch_report = TpUtils::seekDict(kwArgs, "epoch_report", 1);

    EModule model = *this;

    model.train(m_core->m_optimizer);

    dataloader.shuffle();

    for (BatchData data = dataloader.begin(); !data.is_end(); ++data, batch++) {
        ETensor X, y;
        ETensor pred, loss;

        data.get_data(X, y);

        pred = model.__call__(X);
        loss = m_core->m_loss.__call__(pred, y);
        m_core->m_optimizer.zero_grad();
        m_core->m_loss.backward();
        m_core->m_optimizer.step();

        if ((epoch + 1) % epoch_report != 0) continue;

        current += X.len();

        if ((batch_report && batch % batch_report == batch_report - 1) || batch == batch_count - 1) {
            float loss_value = loss.item();
            printf("    loss: %7f  [%5lld/%5lld] (%d secs)", loss_value, current, data_count, (int)(time(NULL) - startTime));
        }
    }
}

void EModule::m_fitValidate(DataLoader dataloader, time_t startTime, VDict kwArgs) {
    int64 data_count = dataloader.data_count();
    int64 batch = 0;

    EModule model = *this;

    model.eval();

    float test_loss = 0;
    float correct = 0;

    nn().set_no_grad();

    for (BatchData data = dataloader.begin(); !data.is_end(); data++, batch++) {
        ETensor X, y;
        data.get_data(X, y);

        ETensor pred = model.__call__(X);

        test_loss += m_core->m_loss.__call__(pred, y).item();

        // correct 계산 방법도 loss 지정 때 함께 설정하여 KaiCppTest::m_test... 통으리할 것,.
        correct += (pred.argmax(1) == y).set_type(VDataType::float32).sum().item();
    }

    nn().unset_no_grad();

    test_loss /= batch;
    correct /= data_count;

    printf("Test Error:  Accuracy: %0.1f%%, Avg loss: %8f (%d secs)", (100 * correct), test_loss, (int)(time(NULL) - startTime));
}
*/

//-----------------------------------------------------------------------------------------------------
// Core part

void EModuleCore::m_setup() {
    m_device = "cpu";
    m_train = true;
}

void EModuleCore::m_delete() {
}

void EModuleCore::m_setup(string sName, string sBuiltin, EModuleType moduleType, VDict kwArgs) {
    m_sName = sName;
    m_sBuiltin = sBuiltin;
    m_moduleType = moduleType;
    m_kwArgs = kwArgs;
}

string EModuleCore::m_desc(int depth, int nth, int64& pm_total, string pos) {
    string indent;

    for (int n = 0; n < depth; n++) indent += "  ";

    string str = indent;

    //str += to_string(m_nn.getEngineObjId(this));

    if (nth >= 0) str += "[" + to_string(nth) + "] ";
    
    VDict dict;

    m_nn.getApiConn()->Module_getModuleInfo(m_hEngineHandle, dict, __FILE__, __LINE__);

    //str += "(#" + to_string((int64)dict["module_id"]) + ": " + (string)dict["name"] + "): " + (string)dict["builtin"];
    string name = (string)dict["name"];
    if (name == "") name = pos;

    str += "(" + name + "): " + (string)dict["builtin"];

    if ((bool)dict["nonterm"]) { //m_moduleType != EModuleType::layer) {
        VList childrenHandles = m_nn.getApiConn()->Module_getChildrenModules(m_hEngineHandle, __FILE__, __LINE__);
        EModuleList children = TpUtils::ListToModuleList(m_nn, childrenHandles);

        if (children.size() > 0) {
            int subnth = 0;

            str += "(\n";
            for (auto& it : children) {
                string childPos = name + "." + to_string(subnth);
                str += it.desc(depth + 1, subnth++, pm_total, childPos) + "\n";
            }

            str += indent + ")";

            VShape temp = dict["inshape"];
            VShape temp2 = dict["outshape"];

            str += indent + temp.desc();
            str += "=>" + temp2.desc() + ")";
        }
    }
    else {
        str += " " + ((VDict)dict["kwargs"]).desc();
        
        VShape temp = dict["inshape"];
        VShape temp2 = dict["outshape"];

        if (temp.size() > 0) {
            str += " " + temp.desc();
            str += "=>" + temp2.desc();
        }

        if (((int64)dict["pmsize"]) > 0) {
            str += " pm:" + to_string((int64)dict["pmsize"]) + " items";
            pm_total += (int64)dict["pmsize"];
            if (0) str += "(" + to_string(pm_total) + ")";
        }
    }

    if (depth == 0) str += " total parameter: " + to_string(pm_total) + " items";
    return str;
}

void EModuleCore::m_saveModel(TpStreamOut& fout) {
    VDict dict;

    m_nn.getApiConn()->Module_getModuleInfo(m_hEngineHandle, dict, __FILE__, __LINE__);

    fout.save_string(dict["builtin"]);
    fout.save_dict(dict["kwargs"]);
    fout.save_bool(dict["nonterm"]);

    if ((bool)dict["nonterm"]) {
        VList childrenHandles = m_nn.getApiConn()->Module_getChildrenModules(m_hEngineHandle, __FILE__, __LINE__);
        EModuleList children = TpUtils::ListToModuleList(m_nn, childrenHandles);
        fout.save_int((int)children.size());
        for (auto& it : children) {
            it.saveModel(fout);
        }
    }
}
