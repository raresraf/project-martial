// By archit-dev, contest: Codeforces Beta Round #63 (Div. 2), problem: (A)
// Young Physicist, Accepted, #, Copy

#include <iostream>

using namespace std;

int main() {
  int n, x, y, z, xsum(0), ysum(0), zsum(0);
  cin >> n;
  while (n--) {
    cin >> x >> y >> z;
    xsum += x;
    ysum += y;
    zsum += z;
  }
  if (xsum == 0 && ysum == 0 && zsum == 0) {
    cout << "YES\n";
  } else {
    cout << "NO\n";
  }
  return 0;
}
