#include <bits/stdc++.h>

using namespace std;

int main() {
  ios_base::sync_with_stdio(false);

  int t;
  cin >> t;

  int c, m, x, smaller, larger;
  int res;

  for (int i = 0; i < t; i++) {
    res = 0;
    cin >> c >> m >> x;

    int maxi = max(c, m);
    int mini = min(c, m);

    x += (maxi - mini);

    if (x >= c || x >= m) {
      cout << min(c, m) << "\n";
      continue;
    }

    maxi = mini;
    maxi -= x;
    mini -= x;

    x += (maxi + mini) / 3;

    cout << x << "\n";
  }
}
