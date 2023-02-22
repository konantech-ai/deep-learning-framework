#include "../objects/tp_audio_file_reader.h"
#include "../objects/tp_audio_file_reader_core.h"
#include "../objects/tp_nn.h"
#include "../objects/tp_tensor.h"
#include "../connect/tp_api_conn.h"
#include "../utils/tp_utils.h"
#include "../utils/tp_exception.h"

EAudioFileReader::EAudioFileReader() { m_core = NULL; }
EAudioFileReader::EAudioFileReader(ENN nn) { m_core = new EAudioFileReaderCore(nn); }
EAudioFileReader::EAudioFileReader(const EAudioFileReader& src) { m_core = src.m_core->clone(); }
EAudioFileReader::EAudioFileReader(EAudioFileReaderCore* core) { m_core = core->clone(); }
EAudioFileReader::~EAudioFileReader() { m_core->destroy(); }
EAudioFileReader& EAudioFileReader::operator =(const EAudioFileReader& src) {
    if (&src != this && m_core != src.m_core) {
        m_core->destroy();
        m_core = src.m_core->clone();
    }
    return *this;
}
bool EAudioFileReader::isValid() { return m_core != NULL; }
void EAudioFileReader::close() { if (this) m_core->destroy(); }
ENN EAudioFileReader::nn() { return m_core ? m_core->m_nn : ENN(); }
EAudioFileReaderCore* EAudioFileReader::getCore() { return m_core; }
EAudioFileReaderCore* EAudioFileReader::cloneCore() { return (EAudioFileReaderCore*)m_core->clone(); }
int EAudioFileReader::meNth() { return m_core->getNth(); }
int EAudioFileReader::meRefCnt() { return m_core->getRefCnt(); }
EAudioFileReaderCore::EAudioFileReaderCore(ENN nn) : VObjCore(VObjType::custom) {
    m_nn = nn;
    m_setup();
}
EAudioFileReaderCore::~EAudioFileReaderCore() {
    m_delete();
}
EAudioFileReaderCore* EAudioFileReader::createApiClone() { return m_core->clone(); }

//-----------------------------------------------------------------------------------------------------
// Capsule part

EAudioFileReader::EAudioFileReader(ENN nn, VDict args) {
    m_core = new EAudioFileReaderCore(nn);
    m_core->m_setup(args);
}

bool EAudioFileReader::addFile(string filePath) {
    return m_core->m_addFile(filePath);
}

ETensor EAudioFileReader::get_fft_spectrums(bool ment) {
    return m_core->m_get_fft_spectrums(ment);
}

//-----------------------------------------------------------------------------------------------------
// Core part

void EAudioFileReaderCore::m_setup() {
}

void EAudioFileReaderCore::m_setup(VDict args) {
    m_freq_in_spectrum = TpUtils::seekDict(args, "freq_in_spectrum", 40); // 하나의 주파수 스펙트럼이 가질 주파수 갯수
    m_spec_interval = TpUtils::seekDict(args, "spec_interval", 256);      // 주파수 스펙트럼들 사이 간격에 해당하는 샘플 수
    m_fft_width = TpUtils::seekDict(args, "fft_width", 2048);             // 각 주파수 스펙트럼 분석에 이용될 오디오 샘플의 수, 2^11에 해당
    m_spec_count = TpUtils::seekDict(args, "spec_count", 500);            // 오디오 특성으로 중앙 부위에서 추출할 스펙트럼 갯수
    m_need_samples = (m_spec_count -1) * m_spec_interval + m_fft_width;  // 추출할 스펙트럼 갯수와 스펙트럼 간의 간격, 스펙트럼 내부 길이를 고려해 계산된 필요 샘플수

    m_nTooShort = 0;
    m_nBadFormat = 0;
}

bool EAudioFileReaderCore::m_addFile(string filePath) {

    try {
        WaveInfoHeader wav_info;
        TpUtils::read_wav_file_header(filePath, &wav_info); // 주의: 다양한 유형의 오디오가 이 함수 내부에서 걸러져 throw 처리됨

        int64 sample_steps = (wav_info.Subchunk2Size * 8) / (wav_info.NumChannels * wav_info.BitsPerSample);      // 파일에 담긴 샘플 갯수
        
        if (sample_steps < m_need_samples) {
            m_nTooShort++;
            throw 1;
        }

        int64 start_pos = wav_info.Subchunk2Offset;
        int64 start_gap_samples = (sample_steps - m_need_samples) / 2;
        int64 bytes_per_sample = sizeof(short) * 2;

        int64 offset = start_pos + start_gap_samples * bytes_per_sample; // file_load_info["offset"];

        m_filePaths.push_back(filePath);
        m_offsets.push_back(offset);

        return true;
    }
    catch (...) {
        m_nBadFormat++;
        return false;
    }
}

ETensor EAudioFileReaderCore::m_get_fft_spectrums(bool ment) {
    int64 file_count = (int64) m_filePaths.size();
    
    VShape wshape{ file_count, m_need_samples, 1 };
    ETensor wave(m_nn, wshape, VDataType::float32);

    if (ment) printf("[CsvmapAudioDataset] 스펙트럼 분석에 필요한 조사된 오디오 파일들의 샘플들을 읽는 중입니다.\n");

    for (int64 n = 0; n < file_count; n++) {
        string filepath = m_filePaths[n];

        FILE* fid = fopen(filepath.c_str(), "rb");
        fseek(fid, (int)m_offsets[n], SEEK_SET);

        //printf("EP1: m_fft_width = %lld\n", m_fft_width);
        //printf("EP1: m_need_samples = %lld\n", m_need_samples);

        float* pWaveBuf = wave.float_ptr() + n * m_need_samples;

        for (int64 nd = 0; nd < m_need_samples; nd++) {
            short word[2];
            if (fread(word, sizeof(short), 2, fid) != 2) TP_THROW(VERR_CONTENT_DATASET); // stereo 2 channels
            pWaveBuf[nd] = word[0] / 32768.0f;
        }

        fclose(fid);
    }

    if (ment) printf("[CsvmapAudioDataset] 스펙트럼 분석을 시작합니다.\n");

    ETensor ffts = m_nn.getApiConn()->Util_fft(wave, m_spec_interval, m_freq_in_spectrum, m_fft_width, __FILE__, __LINE__);

    ffts.downloadData();

    if (ment) printf("[CsvmapAudioDataset] 스펙트럼 분석이 완료되었습니다.\n");

    return ffts;
}

void EAudioFileReaderCore::m_delete() {
}
