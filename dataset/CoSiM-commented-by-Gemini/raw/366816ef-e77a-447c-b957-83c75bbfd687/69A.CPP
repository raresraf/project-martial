#include <bits/stdc++.h>
using namespace std;
int main() {
  int i = 0, n, x, y, z, sumx = 0, sumy = 0, sumz = 0;
  cin >> n;
  while (i < n) {
    cin >> x >> y >> z;
    sumx += x;
    sumy += y;
    sumz += z;
    i++;
  }
  if (sumx == 0 && sumy == 0 && sumz == 0)
    cout << "YES";
  else
    cout << "NO";
  return 0;
}
