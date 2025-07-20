#include <bits/stdc++.h>
using namespace std;
int main() {
  int x, y, z;
  int n;
  cin >> n;
  int sumx, sumy, sumz;
  sumx = sumy = sumz = 0;
  while (n--) {
    cin >> x >> y >> z;
    sumx += x;
    sumy += y;
    sumz += z;
  }
  if (sumx || sumy || sumz)
    cout << "NO\n";
  else
    cout << "YES\n";
  return 0;
}