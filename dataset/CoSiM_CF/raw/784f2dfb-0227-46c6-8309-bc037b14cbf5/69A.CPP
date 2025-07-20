// Accepted
#include <bits/stdc++.h>
using namespace std;
int main() {
  int x, y, z;
  int n;
  int sumx = 0, sumy = 0, sumz = 0;
  cin >> n;
  while (n--) {
    cin >> x >> y >> z;
    sumx += x;
    sumy += y;
    sumz += z;
  }
  if (sumx == 0 && sumy == 0 & sumz == 0)
    cout << "YES" << endl;
  else {
    cout << "NO" << endl;
  }
}