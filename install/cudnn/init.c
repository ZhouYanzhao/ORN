#include "TH.h"
#include "luaT.h"

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_Tensor TH_CONCAT_STRING_3(torch.,Real,Tensor)
#define cudnnorn_(NAME) TH_CONCAT_3(cudnnorn_, Real, NAME)

LUA_EXTERNC DLL_EXPORT int luaopen_libcudnnorn(lua_State *L);

int luaopen_libcudnnorn(lua_State *L)
{
  lua_newtable(L);
  lua_pushvalue(L, -1);
  lua_setfield(L, LUA_GLOBALSINDEX, "cudnnorn");
  return 1;
}
