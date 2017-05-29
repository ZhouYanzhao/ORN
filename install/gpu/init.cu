#include "luaT.h"
#include "THC.h"
#include "utils.h"

typedef unsigned long uint64;
typedef unsigned int uint32;
typedef unsigned short uint16;
typedef unsigned char uint8;

#include "ActiveRotatingFilter.cu"
#include "RotationInvariantEncoding.cu"

LUA_EXTERNC DLL_EXPORT int luaopen_libcuorn(lua_State *L);

int luaopen_libcuorn(lua_State *L)
{
  lua_newtable(L);

  cuorn_ARF_init(L);
  cuorn_RIE_init(L);

  return 1;
}
