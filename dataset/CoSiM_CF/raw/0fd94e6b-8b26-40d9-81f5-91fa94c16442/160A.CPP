#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <iostream>
#include <list>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <utility>
#include <vector>
//#include <ext/hash_map>

using namespace std;
using namespace __gnu_cxx;

#define REP(i, n) for (int i = 0; i < int(n); ++i)
#define REPV(i, n) for (int i = (n)-1; (int)i >= 0; --i)
#define FOR(i, a, b) for (int i = (int)(a); i < (int)(b); ++i)

#define FE(i, t)                                                               \
  for (__typeof((t).begin()) i = (t).begin(); i != (t).end(); ++i)
#define FEV(i, t)                                                              \
  for (__typeof((t).rbegin()) i = (t).rbegin(); i != (t).rend(); ++i)

#define two(x) (1LL << (x))
#define ALL(a) (a).begin(), (a).end()

#define pb push_back
#define ST first
#define ND second
#define MP(x, y) make_pair(x, y)

typedef long long ll;
typedef unsigned long long ull;
typedef unsigned int uint;
typedef pair<int, int> pii;
typedef vector<int> vi;
typedef vector<string> vs;

template <class T> void checkmin(T &a, T b) {
  if (b < a)
    a = b;
}
template <class T> void checkmax(T &a, T b) {
  if (b > a)
    a = b;
}
template <class T> void out(T t[], int n) {
  REP(i, n) cout << t[i] << " ";
  cout << endl;
}
template <class T> void out(vector<T> t, int n = -1) {
  for (int i = 0; i < (n == -1 ? t.size() : n); ++i)
    cout << t[i] << " ";
  cout << endl;
}
inline int count_bit(int n) {
  return (n == 0) ? 0 : 1 + count_bit(n & (n - 1));
}
inline int low_bit(int n) { return (n ^ n - 1) & n; }
inline int ctz(int n) { return (n == 0 ? -1 : ctz(n >> 1) + 1); }
int toInt(string s) {
  int a;
  istringstream(s) >> a;
  return a;
}
string toStr(int a) {
  ostringstream os;
  os << a;
  return os.str();
}

const int maxn = 111111;
int n, m;
int a[maxn];

int main() {
  cin >> m;
  int b = maxn;
  REP(i, m) {
    int x;
    scanf(" %d", &x);
    b = min(b, x);
  }
  cin >> n;
  REP(i, n) scanf(" %d", a + i);
  sort(a, a + n);
  int res = 0;
  REP(i, n) res += a[i];
  for (int i = n - 1; i >= b;) {
    i -= b;
    REP(j, 2) if (i >= 0) res -= a[i--];
  }
  cout << res << endl;

  return 0;
}
