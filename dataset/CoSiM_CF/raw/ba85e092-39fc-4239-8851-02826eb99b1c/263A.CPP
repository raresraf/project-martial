#include <bits/stdc++.h>

#define _for(i, a, b) for (int i = (a), _b_ = (b); i < _b_; i++)
#define _fod(i, a, b) for (int i = (a), _b_ = (b); i > _b_; i--)
#define _it(i, v) for (typeof((v).begin()) i = (v).begin(); i != (v).end(); ++i)
#define _all(v) v.begin(), v.end()
#define __(v) memset(v, 0, sizeof(v))

using namespace std;
typedef long long LL;
typedef unsigned long long ULL;
template <typename T> vector<T> &operator+=(vector<T> &v, T x) {
  v.push_back(x);
  return v;
}

void solve() {
  int a;
  int x, y;
  _for(i, 0, 5) {
    _for(j, 0, 5) {
      cin >> a;
      if (a) {
        x = i;
        y = j;
      }
    }
  }
  cout << abs(x - 2) + abs(y - 2) << endl;
}

int main() {
  ios_base::sync_with_stdio(0);
  cin.tie(0);
#ifdef HIEUNV
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  solve();
}
