#include "../utils/tp_math.h"
#include "../objects/tp_tensor.h"
#include "../utils/tp_exception.h"

#ifdef NOT_DEPRECIATED

ETensor Math::evaluate(string expression, VList args) {
    TP_THROW(VERR_NOT_IMPLEMENTED_YET);
}

ETensor Math::linspace(float from, float to, int count, VDict kwArgs) {
    TP_THROW(VERR_NOT_IMPLEMENTED_YET);
}

ETensor Math::sin(ETensor) {
    TP_THROW(VERR_NOT_IMPLEMENTED_YET);
}

ETensor Math::randh(VShape shape, VDict kwArgs) {
    TP_THROW(VERR_NOT_IMPLEMENTED_YET);
}

#endif