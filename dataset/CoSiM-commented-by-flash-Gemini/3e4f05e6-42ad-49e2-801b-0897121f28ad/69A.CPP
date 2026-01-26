#include <iostream>
using namespace std;

int main() {
  int n;
  cin >> n;

  int x[n], y[n], z[n];
  int xf = 0, yf = 0, zf = 0;
  for (int i = 0; i < n; ++i) {
    cin >> x[i] >> y[i] >> z[i];
    xf += x[i];
    yf += y[i];
    zf += z[i];
  }

  if (xf == 0 && yf == 0 && zf == 0)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}
/*
http://codeforces.com/problemset/problem/69/A
*/
