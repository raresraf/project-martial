#include <algorithm>
#include <iomanip>
#include <iostream>
#include <queue>
#include <string>
#include <utility>
#include <vector>

using namespace std;

typedef long long int ll;
typedef pair<int, int> Pii;

const ll MOD = 1000000007;

int main() {
  cin.tie(0);
  ios::sync_with_stdio(false);

  int q;
  cin >> q;
  vector<int> c(q), m(q), x(q);
  for (int i = 0; i < q; i++)
    cin >> c[i] >> m[i] >> x[i];

  vector<int> ans(q);
  for (int t = 0; t < q; t++) {
    if (c[t] <= m[t] && c[t] <= x[t]) {
      ans[t] = c[t];
    } else if (m[t] <= c[t] && m[t] <= x[t]) {
      ans[t] = m[t];
    } else {
      int total = x[t];
      int h = max(c[t], m[t]) - x[t];
      int l = min(c[t], m[t]) - x[t];
      if (h / 2 >= l)
        total += l;
      else {
        total += h - l;
        int r = l - (h - l);
        total += (r / 3) * 2;
        if (r % 3 == 2)
          total += 1;
      }
      ans[t] = total;
    }
  }

  for (auto &y : ans)
    cout << y << endl;

  return 0;
}
