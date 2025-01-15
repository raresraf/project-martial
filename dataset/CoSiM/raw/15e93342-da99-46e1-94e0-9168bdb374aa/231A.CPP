
/*                     */

#include <bits/stdc++.h>
using namespace std;

int main() {
  int t, a, b, c, f = 0;

  cin >> t;

  while (t--) {
    cin >> a >> b >> c;
    if (a == 1 && b == 1 && c == 1 || a == 1 && b == 1 || b == 1 && c == 1 ||
        a == 1 && c == 1)
      f++;
  }
  cout << f << endl;

  return 0;
}
