#include <bits/stdc++.h>
using namespace std;
int main() {
  long long t, a, b, c, x, i;
  while (cin >> t) {
    x = 0;
    for (i = 1; i <= t; i++) {
      cin >> a >> b >> c;
      if (a == 1 && b == 1)
        x++;
      else if (a == 1 && c == 1)
        x++;
      else if (b == 1 && c == 1)
        x++;
    }
    cout << x << endl;
  }
  return 0;
}
