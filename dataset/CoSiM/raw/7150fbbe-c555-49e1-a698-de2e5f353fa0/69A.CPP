#include <bits/stdc++.h>
using namespace std;

int main() {
  int x, y, z;
  int a = 0, b = 0, c = 0;
  int n;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> x >> y >> z;
    a += x;
    b += y;
    c += z;
  }
  if (a == 0 && b == 0 && c == 0)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}
