#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, y, x, z, suma = 0, sumb = 0, sumc = 0;
  cin >> n;
  while (n--) {
    cin >> x >> y >> z;
    suma = suma + x;
    sumb = sumb + y;
    sumc = sumc + z;
  }
  if (suma || sumb || sumc)
    cout << "NO" << endl;
  else
    cout << "YES" << endl;
  return 0;
}
