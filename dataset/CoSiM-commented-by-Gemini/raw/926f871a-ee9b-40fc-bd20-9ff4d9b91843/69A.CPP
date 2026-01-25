#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, x, y, z;
  int x1 = 0;
  int y1 = 0;
  int z1 = 0;
  int total;
  int sum = 0;
  cin >> n;
  for (int i = 1; i <= n; i++) {
    cin >> x >> y >> z;
    x1 = x1 + x;
    y1 = y1 + y;
    z1 = z1 + z;
  }
  if (x1 == 0 && y1 == 0 && z1 == 0) {
    cout << "YES";
  } else {
    cout << "NO";
  }
}
