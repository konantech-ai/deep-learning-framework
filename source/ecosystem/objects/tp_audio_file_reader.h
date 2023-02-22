#pragma once

#include "../utils/tp_common.h"

class EAudioFileReaderCore;
    class EAudioFileReader {
   
    public:
        EAudioFileReader();
        EAudioFileReader(ENN nn);
        EAudioFileReader(const EAudioFileReader& src);
        EAudioFileReader(EAudioFileReaderCore* core);
        virtual ~EAudioFileReader();
        EAudioFileReader& operator =(const EAudioFileReader& src);
        bool isValid();
        void close();
        ENN nn();
        EAudioFileReaderCore* getCore();
        EAudioFileReaderCore* cloneCore();
        int meNth();
        int meRefCnt();
        EAudioFileReaderCore* createApiClone();
    protected:
        EAudioFileReaderCore* m_core;
    public:

    public:
        EAudioFileReader(ENN nn, VDict args);

        bool addFile(string filePath);

        ETensor get_fft_spectrums(bool ment);

        /*
        string getInstName();

        void registUserDefFunc(EAudioFileReader* pInst);

        virtual ETensor forward(int nInst, ETensor x, VDict opArgs);
        virtual ETensor forward(int nInst, ETensorList operands, VDict opArgs);

        virtual ETensor backward(int nInst, ETensor ygrad, ETensor x, VDict opArgs);
        virtual ETensor backward(int nInst, ETensor ygrad, int nth, ETensorList operands, VDict opArgs);
        */

};
