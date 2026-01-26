#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  cin >> n;
  int x, y, z;
  int t1 = 0, t2 = 0, t3 = 0;
  for (int i = 0; i < n; i++) {
    cin >> x >> y >> z;
    t1 += x;
    t2 += y;
    t3 += z;
  }
  if (t1 == 0 && t2 == 0 && t3 == 0) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }
  return 0;
}
