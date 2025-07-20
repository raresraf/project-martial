#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, sumx = 0, sumy = 0, sumz = 0;
  cin >> n;
  for (int i = 1; i <= n; i++) {
    int x, y, z;
    cin >> x >> y >> z;
    sumx += x;
    sumy += y;
    sumz += z;
  }
  if (!sumx && !sumy && !sumz)
    cout << "YES\n";
  else
    cout << "NO\n";
  return 0;
}
