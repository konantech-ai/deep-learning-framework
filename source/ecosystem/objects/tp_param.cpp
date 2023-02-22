#include "../objects/tp_param.h"
#include "../objects/tp_nn.h"
#include "../connect/tp_api_conn.h"
#include "../utils/tp_utils.h"
#include "../utils/tp_exception.h"

#ifdef NOT_DEPRECIATED

Param::Param(ENN nn, ETensor init) : ETensor(nn) {
    TP_THROW(VERR_NOT_IMPLEMENTED_YET);
}

Param::~Param() {
}
#endif