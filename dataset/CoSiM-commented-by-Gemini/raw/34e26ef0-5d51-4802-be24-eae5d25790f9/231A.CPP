#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, i, co = 0, a, b, c, t = 0;
  cin >> n;
  for (i = 1; i <= n; i++) {
    cin >> a >> b >> c;
    if (a == 1)
      co++;
    if (b == 1)
      co++;
    if (c == 1)
      co++;
    if (co == 2 || co == 3) {
      t++;
      co = 0;
    }
    co = 0;
  }
  cout << t << endl;
}
