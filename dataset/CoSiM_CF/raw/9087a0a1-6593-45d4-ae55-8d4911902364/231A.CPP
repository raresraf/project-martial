#include <iostream>
using namespace std;
main() {
  int n, p, v, t, ctr = 0;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> p >> v >> t;
    if (p + v + t >= 2) {
      ctr++;
    }
  }
  cout << ctr;
}
