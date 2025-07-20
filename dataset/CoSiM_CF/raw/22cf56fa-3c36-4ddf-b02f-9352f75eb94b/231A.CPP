#include <iostream>
using namespace std;
int main() {
  int i, j, c = 0;
  cin >> i;
  for (j = 0; j < i; j++) {
    int p, q, r;
    cin >> p >> q >> r;
    if ((p + q + r) >= 2)
      c++;
  }
  cout << c << endl;
  return 0;
}
