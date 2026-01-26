#include <iostream>
using namespace std;
main() {
  int n;
  int sumx = 0, sumy = 0, sumz = 0;
  cin >> n;
  while (n > 0) {
    int x, y, z;
    cin >> x >> y >> z;
    sumx += x;
    sumy += y;
    sumz += z;
    n -= 1;
  }
  if (sumx == 0 && sumy == 0 && sumz == 0)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
}
