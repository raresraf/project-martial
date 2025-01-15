
#include <algorithm>
#include <bitset>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <deque>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <string.h>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace std;
typedef long long LL;
typedef pair<int, int> pii;

#define forup(i, a, b) for (int i = a; i < b; ++i)
#define fordn(i, a, b) for (int i = a; i > b; --i)
#define rep(i, a) for (int i = 0; i < a; ++i)

#define dforup(i, a, b) for (i = a; i < b; ++i)
#define dfordn(i, a, b) for (i = a; i > b; --i)
#define drep(i, a) for (i = 0; i < a; ++i)

#define slenn(s, n)                                                            \
  for (n = 0; s[n] != 13 and s[n] != 0; ++n)                                   \
    ;                                                                          \
  s[n] = 0

#define gi(x) scanf("%d", &x)
#define gl(x) scanf("%lld", &x)
#define gd(x) scanf("%lf", &x)
#define gs(x) scanf("%s", x)
#define printvi(x) rep(i, x.size()) pis(x)
#define pis(x) printf("%d ", x)
#define pin(x) printf("%d\n", x)
#define pls(x) printf("%lld ", x)
#define pln(x) printf("%lld\n", x)
#define pds(x) printf("%.12f ", x)
#define pdn(x) printf("%.12f\n", x)
#define pnl() printf("\n")

#define fs first
#define sc second
#define ll long long
#define pb push_back
#define MOD 1000000007
#define limit 10000005
#define INF 1000000000
#define ull unsigned long long
using namespace std;
ull mod_pow(ull num, ull pow, ull mod) {
  ull test, n = num;
  for (test = 1; pow; pow >>= 1) {
    if (pow & 1)
      test = ((test % mod) * (n % mod)) % mod;
    n = ((n % mod) * (n % mod)) % mod;
  }
  return test; /* note this is potentially lossy */
}
// while((getchar())!='\n'); //buffer clear
ll gcd(ll a, ll b) {
  ll r;
  while (b) {
    r = a % b;
    a = b;
    b = r;
  }
  return a;
}

int main() {
  int n;
  int k;
  string s;
  gi(n);
  gi(k);
  cin >> s;
  while (k--) {
    rep(i, n - 1) {
      if (s[i] == 'B' && s[i + 1] == 'G') {
        s[i] = 'G';
        s[i + 1] = 'B';
        i += 1;
      }
    }
  }
  cout << s << endl;
}
