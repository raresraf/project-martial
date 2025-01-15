#include <bits/stdc++.h>

using namespace std;

int main() {
  int t;
  cin >> t;

  int x, y, z, sumx = 0, sumy = 0, sumz = 0;
  for (int i = 0; i < t; i++) {
    cin >> x >> y >> z;
    sumx += x;
    sumy += y;
    sumz += z;
  }

  if (sumx == 0 && sumy == 0 && sumz == 0)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}
