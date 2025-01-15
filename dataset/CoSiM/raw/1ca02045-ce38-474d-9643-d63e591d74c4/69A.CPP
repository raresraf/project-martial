#include <iostream>

using namespace std;

int main() {
  int n, sumx = 0, sumy = 0, sumz = 0, x, y, z;
  cin >> n;
  // int x[n],y[n],z[n];
  for (int i = 0; i < n; i++) {
    cin >> x >> y >> z;
    sumx += x;
    sumy += y;
    sumz += z;
  }
  if (sumx == sumy && sumy == sumz && sumz == 0)
    cout << "YES";
  else
    cout << "NO";
}
