#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, x1, y1, z1;
  int xSum = 0, ySum = 0, zSum = 0;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> x1 >> y1 >> z1;
    xSum += x1;
    ySum += y1;
    zSum += z1;
  }
  if (xSum == 0 && ySum == 0 && zSum == 0) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }
  return 0;
}
