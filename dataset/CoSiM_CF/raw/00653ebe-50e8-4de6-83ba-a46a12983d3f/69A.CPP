#include <bits/stdc++.h>

using namespace std;

int main() {
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  int n;
  cin >> n;
  int x, y, z;
  int xSum = 0;
  int ySum = 0;
  int zSum = 0;

  while (n > 0) {
    cin >> x >> y >> z;
    xSum = xSum + x;
    ySum = ySum + y;
    zSum = zSum + z;
    n--;
  }

  if (xSum == 0 && ySum == 0 && zSum == 0) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }
  return 0;
}