#include <iostream>
using namespace std;

int main() {
  int i, n, a, b, c, ct = 0, ct1;
  cin >> n;
  for (i = 1; i <= n; i++) {
    cin >> a >> b >> c;
    ct1 = 0;
    if (a == 1)
      ct1++;
    if (b == 1)
      ct1++;
    if (c == 1)
      ct1++;
    if (ct1 > 1)
      ct++;
  }
  cout << ct;
  return 0;
}
