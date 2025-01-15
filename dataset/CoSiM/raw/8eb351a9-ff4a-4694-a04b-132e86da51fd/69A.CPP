#include <bits/stdc++.h>
using namespace std;

int main() {
  int i, n, x, y, z, sox = 0, soy = 0, soz = 0;

  cin >> n;
  for (i = 1; i <= n; i++) {
    cin >> x >> y >> z;
    sox += x;
    soy += y;
    soz += z;
  }
  if (sox == 0 && soy == 0 && soz == 0)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}
