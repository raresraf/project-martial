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

bool check(int a) {
  if (a < 4)
    return false;
  bool check = true;
  while (a > 0) {
    if (a % 10 != 7 && a % 10 != 4)
      return false;
    a /= 10;
  }
  return check;
}

void solve() {
  string s;
  cin >> s;
  int c = 0;
  _for(i, 0, s.length()) {
    if (s[i] == '4' || s[i] == '7')
      c++;
  }
  if (check(c))
    cout << "YES";
  else
    cout << "NO";
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
