#include "TH.h"
#include "luaT.h"

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_Tensor TH_CONCAT_STRING_3(torch.,Real,Tensor)
#define orn_(NAME) TH_CONCAT_3(orn_, Real, NAME)

typedef unsigned long uint64;
typedef unsigned int uint32;
typedef unsigned short uint16;
typedef unsigned char uint8;

#include "generic/ActiveRotatingFilter.c"
#include "THGenerateFloatTypes.h"

#include "generic/RotationInvariantEncoding.c"
#include "THGenerateFloatTypes.h"

LUA_EXTERNC DLL_EXPORT int luaopen_liborn(lua_State *L);

int luaopen_liborn(lua_State *L)
{
  lua_newtable(L);
  lua_pushvalue(L, -1);
  lua_setfield(L, LUA_GLOBALSINDEX, "orn");

  orn_FloatARF_init(L);
  orn_DoubleARF_init(L);

  orn_FloatRIE_init(L);
  orn_DoubleRIE_init(L);

  return 1;
}
