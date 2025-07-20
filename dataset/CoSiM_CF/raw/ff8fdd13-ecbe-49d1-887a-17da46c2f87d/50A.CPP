#include <iostream>
using namespace std;
int main() {
  int m, n, i, j, r, c;
  while (cin >> m >> n) {
    r = m * (n / 2);
    if (n % 2 != 0) {
      j = n % 2;
      c = j * (m / 2);
      cout << r + c << endl;
    } else {
      cout << r << endl;
    }
  }
  return 0;
}
