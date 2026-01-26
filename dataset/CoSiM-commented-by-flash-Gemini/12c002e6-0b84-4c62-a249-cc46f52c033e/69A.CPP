#include <bits/stdc++.h>
using namespace std;

int main() {
  int x, y, z, n, xsum(0), ysum(0), zsum(0);

  cin >> n;

  while (n--) {
    cin >> x >> y >> z;
    xsum += x;
    ysum += y;
    zsum += z;
  }

  if (xsum == 0 && ysum == 0 && zsum == 0) {
    cout << "YES" << endl;
  } else
    cout << "NO" << endl;

  return 0;
}