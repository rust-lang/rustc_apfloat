/* HACK(eddyb) we want standard `assert`s to work, but `NDEBUG` also controls
   unrelated LLVM facilities that are spread all over the place and it's harder
   to compile all of them, than do this workaround where we shadow `assert.h`. */

#undef NDEBUG
#include_next <assert.h>
#define NDEBUG
