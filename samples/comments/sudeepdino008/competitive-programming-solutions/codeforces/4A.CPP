
#define DEBUG  // comment when you have to disable all debug macros.
#define NDEBUG // comment when all assert statements have to be disabled.
#include <algorithm>
#include <bitset>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <functional>
#include <iomanip>
#include <iostream>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <sys/time.h>
#include <utility> //std::pair
#include <vector>
#define tr(c, i) for (auto i = (c).begin(); i != (c).end(); i++)
#define present(c, x) ((c).find(x) != (c).end())
#define all(x) x.begin(), x.end()
#define pb push_back
#define mp make_pair
#define log2(x) (log(x) / log(2))
#define ARRAY_SIZE(arr) (1 [&arr] - arr)
#define lld long long int
#define MOD 1000000007
using namespace std;

#ifdef DEBUG
#define debug(args...)                                                         \
  {                                                                            \
    dbg, args;                                                                 \
    cerr << endl;                                                              \
  }
#else
#define debug(args...) // Just strip off all debug tokens
#endif

struct debugger {
  template <typename T> debugger &operator,(const T &v) {
    cerr << v << " ";
    return *this;
  }

} dbg;

int main() {
  int w;
  cin >> w;
  if (w == 2 || w & 1)
    cout << "NO\n";
  else
    cout << "YES\n";
}
