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
  int n, t;
  cin >> n >> t;
  string s;
  cin >> s;
  while (t--) {
    _for(i, 1, s.length()) {
      if (s[i] == 'G' && s[i - 1] == 'B') {
        swap(s[i], s[i - 1]);
        i++;
      }
    }
  }
  cout << s;
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
