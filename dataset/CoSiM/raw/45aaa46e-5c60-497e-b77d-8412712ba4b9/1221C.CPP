#include <bits/stdc++.h>
using namespace std;

int main() {

#ifdef d
  freopen("input.txt", "r", stdin);
#endif

  int t;
  cin >> t;
  while (t--) {
    int c, m, x;
    cin >> c >> m >> x;
    int y = min(c, m);
    int z = min(y, (c + m + x) / 3);
    cout << z << "\n";
  }

  return 0;
}
