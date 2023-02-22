#pragma once

#include "../utils/tp_common.h"
#include "../objects/tp_nn.h"

class EAudioFileReaderCore : public VObjCore {
protected:
    friend class EAudioFileReader;
protected:
    EAudioFileReaderCore(ENN nn);
    ~EAudioFileReaderCore();
    EAudioFileReaderCore* clone() { return (EAudioFileReaderCore*)clone_core(); }

protected:
    void m_setup();
    void m_setup(VDict args);
    void m_delete();

    bool m_addFile(string filePath);
    
    ETensor m_get_fft_spectrums(bool ment);

protected:
    ENN m_nn;

    bool m_bBadFile;

    int64 m_freq_in_spectrum;  // 하나의 주파수 스펙트럼이 가질 주파수 갯수
    int64 m_fft_width;         // 각 주파수 스펙트럼 분석에 이용될 오디오 샘플의 수, 2의 거듭제곱
    int64 m_spec_interval;     // 주파수 스펙트럼들 사이 간격에 해당하는 샘플 수
    int64 m_spec_count;        // 오디오 특성으로 중앙 부위에서 추출할 스펙트럼 갯수
    int64 m_need_samples;      // 추출할 스펙트럼 갯수와 스펙트럼 간의 간격, 스펙트럼 내부 길이를 고려해 계산된 필요 샘플수

    vector<string> m_filePaths;
    vector<int64> m_offsets;

    int m_nTooShort;
    int m_nBadFormat;

protected:
    //void m_setup(string sName, string sBuiltin, VDict kwArgs);

protected:
    //string m_sBuiltin;
    //string m_sName;
};
