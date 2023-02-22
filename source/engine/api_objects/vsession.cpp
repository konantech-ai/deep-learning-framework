#include "../api_objects/vsession_core.h"
#include "../api_objects/vsession.h"
#include "../api_objects/vfunction.h"
#include "../api_objects/vmodule.h"
#include "../api_objects/vmodule_core.h"
#include "../local_objects/vdevicemanager.h"
#include "../local_objects/vcbbackinfo.h"
#include "../local_objects/vcbbackslot.h"
#include "../support/vmath.h"

//-----------------------------------------------------------------------------------------------------
// 기본 구성

string VSession::ms_sEngineVersion = "0.0.1";

int VSessionCore::ms_nCheckCode = 64179182;

VSession::VSession() {
    m_core = NULL;
}

VSession::VSession(const VSession& src) {
    m_core = src.m_core->clone();
}

VSession::VSession(VDict kwArgs) {
    m_core = new VSessionCore();
    m_core->m_nCheckCode = VSessionCore::ms_nCheckCode;
    m_core->m_propDict = kwArgs;
    m_core->m_deviceManager = VDeviceManager("devman", {});
    m_core->m_hyperManager = VHyperManager("hyper_man", {});
    m_core->m_bNeedForwardCallbeck = false;
    m_core->m_bNeedBackwardCallbeck = false;
    m_core->m_funcCbHadlerInfo.isValid = false;
}

VSession::VSession(VHSession hSession) {
    m_core = NULL;
    VSessionCore* core = (VSessionCore*)hSession;
    if (core == NULL || core->m_nCheckCode != VSessionCore::ms_nCheckCode) VP_THROW(VERR_INVALID_CORE);
    m_core = ((VSessionCore*)hSession)->clone();
}

VSession::~VSession() {
    m_core->destroy();
}

int VSession::getIdForHandle(VHandle handle) {
    VObjCore* pCore = (VObjCore*)handle;
    return pCore ? pCore->getNth() : -1;
}

void VSession::seedRandom(int64 rand_seed) {
    ::srand((unsigned int)rand_seed);
    VMath::seed_random(rand_seed);
}

VSession& VSession::operator =(const VSession& src) {
    if (&src != this && m_core != src.m_core) {
        m_core->destroy();
        m_core = src.m_core->clone();
    }
    return *this;
}

bool VSession::operator ==(const VSession& src) const {
    return m_core == src.m_core;
}

bool VSession::operator !=(const VSession& src) const {
    return m_core != src.m_core;
}

VHSession VSession::cloneCore() {
    return (VHSession)m_core->clone_core();
}

VHSession VSession::cloneHandle() {
    return (VHSession)m_core->clone_handle();
}

void VSession::closeHandle() {
    m_core->destroy_handle();
}

void VSession::registCallbackSlot(VCbBackSlot slot) {
    m_core->m_registCallbackSlot(slot);
}

void VSession::closeObjectInfo() {
    m_core->m_closeObjectInfo();
}

void VSession::registUserDefinedFunction(string sName, VFunction function) {
    m_core->m_userFuncMap[sName] = function;
}

void VSession::setFunctionCbHandler(void* pCbAux, VCbForwardFunction* pFuncForward, VCbBackwardFunction* pFuncBackward, VCbClose* pCbClose) {
    m_core->m_funcCbHadlerInfo.isValid = true;
    m_core->m_funcCbHadlerInfo.m_pFuncCbAux = pCbAux;
    m_core->m_funcCbHadlerInfo.m_pFuncCbForward = pFuncForward;
    m_core->m_funcCbHadlerInfo.m_pFuncCbBackward = pFuncBackward;
    m_core->m_funcCbHadlerInfo.m_pFuncCbClose = pCbClose;
}

VFuncCbHandlerInfo& VSession::getFunctionCbHandlerInfo() {
    if (!m_core->m_funcCbHadlerInfo.isValid) VP_THROW(VERR_INVALID_FUNCTION_CALLBACK_HANDLER_INFO);
    return m_core->m_funcCbHadlerInfo;
}

bool VSession::lookupUserDefinedFunctions(string opCode, VValue* pFunction) {
    return m_core->m_lookupUserDefinedFunctions(opCode, pFunction);
}

VSession::operator VHSession() {
    return (VHSession)m_core->cloneHandle();
}

VSessionCore::VSessionCore() : VObjCore(VObjType::Session) {
    m_noGrad = false;
    if (0) m_noTracer = true;
    m_noTracer = false;

    m_customModuleExecCbFunc = NULL;
    m_pCustomModuleExecInst = NULL;
    m_pCustomModuleExecAux = NULL;
}

VSessionCore::~VSessionCore() {
    m_nCheckCode = 0;
}

string VSession::getVersion() {
    return ms_sEngineVersion;
}

bool VSession::getNoGrad() {
    return m_core->m_noGrad;
}

bool VSession::getNoTracer() {
    return m_core->m_noTracer;
}

void VSession::setNoGrad(bool noGrad) {
    m_core->m_noGrad = noGrad;
}

void VSession::setNoTracer(bool noTracer) {
    m_core->m_noTracer = noTracer;
}

VDeviceManager VSession::device_man() {
    return m_core->m_deviceManager;
}

VHyperManager VSession::hyper_man() {
    return m_core->m_hyperManager;
}

void VSession::registMacro(string macroName, VModule module, VDict kwArgs) {
    module.setMacroArgs(kwArgs);
    m_core->m_macros[macroName] = module.cloneCore();
}

VModule VSession::getMacro(string macroName) {
    if (m_core->m_macros.find(macroName) == m_core->m_macros.end()) {
        printf("MP1: macroName = %s\n", macroName.c_str());
        VP_THROW1(VERR_MACRO_UNREGISTERED, macroName);
    }
    return VModule(*this, (VHModule)m_core->m_macros[macroName]);
}

void VSession::invokeCallback(
    VModuleCore* pStarter, VCbItem item, string name, bool train, int nDevice, bool bPre,
    bool noGrad, VTensorDict xs, VTensorDict ys, VTensorDict sideTerms, VParameters params, VExecTracer tracer) {
    m_core->m_invokeCallback(pStarter, item, name, train, nDevice, bPre, noGrad, xs, ys, sideTerms, params, tracer);
}

//-----------------------------------------------------------------------------------------------------
// 캡슐 영역 확장 코드

void VSession::RegistCustomModuleExecFunc(VCbCustomModuleExec* pFunc, void* pInst, void* pAux) {
    m_core->m_customModuleExecCbFunc = pFunc;
    m_core->m_pCustomModuleExecInst = pInst;
    m_core->m_pCustomModuleExecAux = pAux;
}

void VSession::RegistFreeReportBufferFunc(VCbFreeReportBuffer* pFunc, void* pInst, void* pAux) {
    m_core->m_freeReportBufferCbFunc = pFunc;
    m_core->m_pFreeReportBufferInst = pInst;
    m_core->m_pFreeReportBufferAux = pAux;
}

VCbCustomModuleExec* VSession::getCustomModuleExecCbFunc(void** ppInst, void** ppAux) {
    if (ppInst) *ppInst = m_core->m_pCustomModuleExecInst;
    if (ppAux) *ppAux = m_core->m_pCustomModuleExecAux;

    return m_core->m_customModuleExecCbFunc;
}

VCbFreeReportBuffer* VSession::getFreeReportBufferCbFunc(void** ppInst, void** ppAux) {
    if (ppInst) *ppInst = m_core->m_pFreeReportBufferInst;
    if (ppAux) *ppAux = m_core->m_pFreeReportBufferAux;

    return m_core->m_freeReportBufferCbFunc;
}

int VSession::addForwardCallbackHandler(VCbForwardModule* pCbFunc, VCbClose* pCbClose, VDict filters, VDict instInfo) {
    if (filters.find("name") == filters.end()) VP_THROW(VERR_INVALID_DICT_KEY);

    VCbItem cbItem(*this, pCbFunc, pCbClose, filters, instInfo);

    m_core->m_cbForwardItemMap[cbItem.getNth()] = cbItem;
    m_core->m_bNeedForwardCallbeck = true;

    return cbItem.getNth();
}

int VSession::addBackwardCallbackHandler(VCbBackwardModule* pCbFunc, VCbClose* pCbClose, VDict filters, VDict instInfo) {
    VCbItem cbItem(*this, pCbFunc, pCbClose, filters, instInfo);

    m_core->m_cbBackwardItemMap[cbItem.getNth()] = cbItem;
    m_core->m_bNeedBackwardCallbeck = true;

    return cbItem.getNth();
}

void VSession::removeCallbackHandler(int nId) {
    if (m_core->m_cbForwardItemMap.find(nId) != m_core->m_cbForwardItemMap.end()) {
        m_core->m_cbForwardItemMap.erase(nId);
        m_core->m_bNeedForwardCallbeck = (m_core->m_cbForwardItemMap.size() > 0);
    }
    else if (m_core->m_cbBackwardItemMap.find(nId) != m_core->m_cbBackwardItemMap.end()) {
        m_core->m_cbBackwardItemMap.erase(nId);
        m_core->m_bNeedBackwardCallbeck = (m_core->m_cbBackwardItemMap.size() > 0);
    }
    else {
        VP_THROW(VERR_INVALID_MAP_KEY);
    }
}

bool VSession::needCallback() {
    return m_core->m_bNeedForwardCallbeck || m_core->m_bNeedBackwardCallbeck;
}

void VSession::invokeMatchingCallbacks(
        VModuleCore* pStarter, string name, VTensorDict xs, VTensorDict ys, VTensorDict sideTerms, VParameters params,
        bool train, bool noGrad, int nDevice, bool bPre, VCbBackInfo cbInfo, VExecTracer tracer) {
    printf("NP1\n");
    if (m_core->m_bNeedForwardCallbeck) {
        for (auto& item : m_core->m_cbForwardItemMap) {
            int filtered = m_core->m_filterCheck(item.second, name, train, nDevice, bPre);
            if (filtered != 0) continue;
            m_core->m_invokeCallback(pStarter, item.second, name, train, nDevice, bPre, noGrad, xs, ys, sideTerms, params, tracer);
        }
    }
    printf("NP2\n");

    if (m_core->m_bNeedBackwardCallbeck && train) {
        if (name == "conv1") {
            int nnn = 0;
        }

        for (auto& item : m_core->m_cbBackwardItemMap) {
            int filtered = m_core->m_filterCheck(item.second, name, nDevice);
            if (filtered != 0) continue;
            cbInfo.addCbRequestSlot(pStarter, item.second, name, nDevice, bPre, xs, ys, sideTerms, params);
        }
    }
    printf("NP3\n");
}

int VSessionCore::m_filterCheck(VCbItem item, string name, bool train, int nDevice, bool bPre) {
    VDict filters = item.getFilters();

    if (filters.find("name") == filters.end()) return 1;
    if (!((VList)filters["name"]).find_string(name)) return 2;

    for (auto& filter : filters) {
        VList noms = filter.second;

        if (filter.first == "mode") {
            if (train && !noms.find_string("train")) return 3;
            if (!train && !noms.find_string("test")) return 4;
        }
        else if (filter.first == "device") {
            if (noms.find(nDevice) == noms.end()) return 7;
        }
        else if (filter.first == "phase") {
            if (bPre && !noms.find_string("pre")) return 8;
            if (!bPre && !noms.find_string("post")) return 9;
        }
    }

    return 0;
}

int VSessionCore::m_filterCheck(VCbItem item, string name, int nDevice) {
    VDict filters = item.getFilters();

    if (filters.find("name") == filters.end()) return 1;
    if (!((VList)filters["name"]).find_string(name)) return 2;

    for (auto& filter : filters) {
        VList noms = filter.second;

        if (filter.first == "device") {
            if (noms.find(nDevice) == noms.end()) return 7;
        }
    }

    return 0;
}

void VSessionCore::m_invokeCallback(
    VModuleCore* pStarter, VCbItem item, string name, bool train, int nDevice, bool bPre,
    bool noGrad, VTensorDict xs, VTensorDict ys, VTensorDict sideTerms, VParameters params, VExecTracer tracer) {
    VDict instInfo = item.getInstInfo();

    // L#2170 참조계수가 1증가, 1증가, 2증가, 이후 증가 없음???
    instInfo["#data_idx"] = pStarter->getDataIdx(nDevice);

    VDict statusInfo;

    statusInfo["name"] = name;
    statusInfo["mode"] = train ? "train" : "test";
    statusInfo["device"] = nDevice;
    statusInfo["phase"] = bPre ? "pre" : "post";
    statusInfo["no_grad"] = noGrad;
    statusInfo["direction"] = "forward";

    VList names = { "input", "output", "sideterm", "param" };

    VTensorDict tensors[4];
    
    tensors[0] = xs;
    tensors[1] = ys;
    tensors[2] = sideTerms;
    
    //tensors[3] = params.getWeightDict();
    VList _names; // ignore
    params.getWeights(_names, tensors[3], false);

    //VDict tensorDict = vutils.toDictExternal(tensors, names);
    VDict tensorDict = vutils.toDictInternal(tensors, names);   // ㅣㄷ마 qkftod rksmdtjd dlTdma

    VCbForwardModule* pCbFunc = (VCbForwardModule*)item.getCbFunc();
    VCbClose* pCbClose = (VCbClose*)item.getCbClose();

    tracer.addInvokeForwardCallback(pCbFunc, pCbClose, instInfo, statusInfo, tensorDict);

    extern VDict V_invokeModuleForwardCallback(VHSession hSession, VCbForwardModule* pCbFunc, VCbClose * pCbClose, VDict instInfo, VDict statusInfo, VDict tensorDict);

    // 콜백 반값 result는 아직 특별한 용도가 없지만 차후 확장에 대비해 전달받을 수 있게 한다.
    VDict result = V_invokeModuleForwardCallback((int64)this, pCbFunc, pCbClose, instInfo, statusInfo, tensorDict);

    // 아래 줄은 메모리 이중 반납 문제로 문제를 일으킬 소지가 있음, 유사 사례 발견됨, 문제 발생 확인시 삭제
    vutils.freeDictInternal(tensorDict);
}

void VSessionCore::m_registCallbackSlot(VCbBackSlot slot) {
    m_callbackSlots.push_back(slot);
}

bool VSessionCore::m_lookupUserDefinedFunctions(string opCode, VValue* pFunction) {
    if (m_userFuncMap.find(opCode) == m_userFuncMap.end()) return false;
    
    //*pFunction = (VObjCore*)m_userFuncMap[opCode].cloneCore();
    *pFunction = (int64)m_userFuncMap[opCode].getCore(); // 
    return true;
}

void VSessionCore::m_closeObjectInfo() {
    for (auto& it : m_callbackSlots) {
        it.close();
    }

    m_cbForwardItemMap.clear();
    m_cbBackwardItemMap.clear();
    m_callbackSlots.clear();

    m_userFuncMap.clear();
}

void VSession::SetLastError(VException ex) {
    m_core->m_lastError = ex;
}

VRetCode VSession::GetLastErrorCode() {
    return m_core->m_lastError.GetErrorCode();
}

VList VSession::GetLastErrorMessageList() {
    return m_core->m_lastError.GetErrorMessageList();
}

VTensor VSession::util_fft(VTensor x, int64 spec_interval, int64 freq_in_spectrum, int64 fft_width) {
    VExecTracer tracer; // dummy, 처리 과정 레코딩 없음

    if (x.device() < 0) {
        x = x.toDevice(0, tracer);
    }

    if (0) x.dump("input");

    VShape xshape = x.shape();

    int64 batch_size = xshape[0];
    int64 samples_in_data = xshape[1];

    int64 step_cnt = (samples_in_data - fft_width) / spec_interval + 1;

    VShape yshape{ batch_size, step_cnt, freq_in_spectrum };
    VTensor y(*this, yshape, VDataType::float32, 0);

    int64 nBufSizePerBatch = sizeof(float) * step_cnt * fft_width * 4;
    int64 piece_size = m_core->m_deviceManager.getMaxBatchSize(nBufSizePerBatch);

    if (piece_size > 0) {
        if (piece_size > batch_size) piece_size = batch_size;

        VShape bshape{ piece_size, step_cnt, fft_width, 2 };
        VShape fshape{ piece_size, step_cnt, freq_in_spectrum };

        VTensor buf1(*this, bshape, VDataType::float32, 0);
        VTensor buf2(*this, bshape, VDataType::float32, 0);

        float* pbuf1 = buf1.float_ptr();
        float* pbuf2 = buf2.float_ptr();

        int64 slice_from = 0;
        int64 rest_size = batch_size;

        while (rest_size > 0) {
            int64 slice_size = (rest_size >= piece_size) ? piece_size : rest_size;

            int64 bsize = slice_size * step_cnt * fft_width * 2;
            int64 fsize = slice_size * step_cnt * freq_in_spectrum;

            float* pwave = x.float_ptr() + slice_from * samples_in_data;
            float* pffts = y.float_ptr() + slice_from * step_cnt * freq_in_spectrum;

            VMath::fft_wave_to_complex(pbuf1, pwave, bsize, spec_interval, step_cnt, fft_width, samples_in_data);

            if (0) buf1.dump("complex");

            float* psrc = m_fft_core_split(pffts, pbuf1, pbuf2, fft_width, bsize);

            if (0) {
                if (psrc == pbuf1) buf1.dump("fft");
                else buf2.dump("fft");
            }

            VMath::fft_complex_to_abs_mean(pffts, psrc, fsize, fft_width, freq_in_spectrum);

            if (0) y.dump("result");

            slice_from += slice_size;
            rest_size -= slice_size;
        }
    }
    else {
        for (int64 n = 0; n < batch_size; n++) {
            int64 nBufSizePerBatch = sizeof(float) * fft_width * 4;
            int64 piece_size = m_core->m_deviceManager.getMaxBatchSize(nBufSizePerBatch);

            if (piece_size <= 0) VP_THROW(VERR_SIZE_BUFFER);

            if (piece_size > step_cnt) piece_size = step_cnt;

            VShape bshape{ piece_size, fft_width, 2 };
            VShape fshape{ piece_size, freq_in_spectrum };

            VTensor buf1(*this, bshape, VDataType::float32, 0);
            VTensor buf2(*this, bshape, VDataType::float32, 0);

            float* pbuf1 = buf1.float_ptr();
            float* pbuf2 = buf2.float_ptr();

            int64 bsize = bshape.total_size();
            int64 fsize = fshape.total_size();

            int64 slice_from = 0;
            int64 rest_size = step_cnt;

            while (rest_size > 0) {
                int64 slice_size = (rest_size >= piece_size) ? piece_size : rest_size;

                float* pwave = x.float_ptr() + n * samples_in_data + slice_from * spec_interval;
                float* pffts = y.float_ptr() + n * step_cnt * freq_in_spectrum + slice_from * freq_in_spectrum;

                VMath::fft_wave_to_complex(pbuf1, pwave, bsize, spec_interval, step_cnt, fft_width, samples_in_data);

                if (0) buf1.dump("complex");

                float* psrc = m_fft_core_split(pffts, pbuf1, pbuf2, fft_width, bsize);

                if (0) {
                    if (psrc == pbuf1) buf1.dump("fft");
                    else buf2.dump("fft");
                }

                VMath::fft_complex_to_abs_mean(pffts, psrc, fsize, fft_width, freq_in_spectrum);

                slice_from += slice_size;
                rest_size -= slice_size;
            }

            if (0) y.dump("result");
        }
    }

    return y;
}

float* VSession::m_fft_core_split(float* pffts, float* pbuf1, float* pbuf2, int64 fft_width, int64 bsize) {
    float* psrc = pbuf1;
    float* pdst = pbuf2;

    int64 mb_size = bsize / (fft_width * 2);
    int64 sp_size = mb_size;
    int64 ssize = bsize;

    int64 step = 1;

    while (step < fft_width) {
        step = step * 2;

        VMath::fft_step_split(pdst, psrc, ssize, fft_width, step);

        float* ptmp = pdst;
        pdst = psrc;
        psrc = ptmp;
    }

    return psrc;
}

VDict VSession::getLeakInfo(bool sessionOnly) {
    extern void dumpObjectUsage(string title);
    VP_THROW(VERR_NOT_IMPLEMENTED_YET);
}

//-----------------------------------------------------------------------------------------------------
// 코어 영역 확장 코드
