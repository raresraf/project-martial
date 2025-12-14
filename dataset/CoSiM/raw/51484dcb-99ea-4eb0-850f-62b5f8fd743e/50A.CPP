#include <bits/stdc++.h>
using namespace std;
main() {
  int n, m, a = 2, x, y;
  cin >> m >> n;
  if (n <= 2 || m <= 2) {
    cout << n << endl;
  } else {
    if (n % a == 0) {
      x = n / a;
    } else {
      x = n / a + 1;
    }
    if (m % a == 0) {
      y = m / a;
    } else {
      y = m / a + 1;
    }
    cout << x * y << endl;
  }
}
