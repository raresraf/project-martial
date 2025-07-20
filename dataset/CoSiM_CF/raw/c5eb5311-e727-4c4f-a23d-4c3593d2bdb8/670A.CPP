#include <bits/stdc++.h>
using namespace std;
#define fi first
#define se second
#define mp make_pair
#define rep(i, n) for (int i = 0; i < (int)n; ++i)
#define repr(i, n) for (int i = (int)n; i > -1; --i)
#define REP(i, start, end, step)                                               \
  for (int i = (int)start; i < (int)end; i += (int)step)
#define REPR(i, start, end, step)                                              \
  for (int i = (int)start; i > (int)end; i += (int)step)
#define repc(i, c) for (auto i = c.begin(); i != c.end(); i++)
#define repcr(i, c) for (auto i = c.rbegin(); i != c.rend(); i++)
#define lowbit(x) ((x) & (-x))
typedef long long ll;
typedef unsigned long long ull;
// global variables
const int MAXN = 102;
int N, A[MAXN];

//-------------function-starts---------------------
//-------------function-ends-----------------------

void solve() {
  int cnt[102] = {0};
  rep(i, N) cnt[A[i]]++;
  int rest = 2;
  int res = 0, cur = 0;
  rep(i, *max_element(A, A + N) + 1) {
    cur = i + 1;
    if (cnt[i] < rest) {
      if (cnt[i] == 0) {
        res += rest * i;
        cout << res << endl;
        return;
      } else {
        res += i;
        rest--;
      }
    }
  }
  cout << res + rest * cur << endl;
}

int main() {
  int T;
  cin >> T;
  while (T--) {
    // read params to global variables
    cin >> N;
    rep(i, N) cin >> A[i];
    solve();
  }
  return 0;
}