#include <cmath>
#include <iostream>
using namespace std;

int main() {
  int n, a, x, y, z;
  cin >> n;
  x = 0;
  y = 0;
  z = 0;
  n = n * 3;
  while (n > 0) {
    cin >> a;
    if (n % 3 == 0) {
      x = x + a;
    } else if (n % 3 == 1) {
      y = y + a;
    } else {
      z = z + a;
    }
    n = n - 1;
  }

  if (x == 0 && y == 0 && z == 0) {
    cout << "YES";
  } else {
    cout << "NO";
  }
  return 0;
}
