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
  int n;
  cin >> n;
  vector<int> arr;
  int a, all = 0;
  _for(i, 0, n) {
    cin >> a;
    all += a;
    arr.push_back(a);
  }
  sort(_all(arr));
  int rs = 0, sum = 0;
  _fod(i, arr.size() - 1, -1) {
    if (sum <= all / 2) {
      sum += arr[i];
      rs++;
    }
  }
  cout << rs << endl;
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
