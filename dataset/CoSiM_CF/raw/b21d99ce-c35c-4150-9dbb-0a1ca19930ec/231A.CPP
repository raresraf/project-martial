#include <iostream>
using namespace std;

int main() {
  int n, p, v, t, i, sum = 0;
  cin >> n;
  for (i = 0; i < n; i++) {
    cin >> p >> v >> t;
    if (p + v + t >= 2) {
      sum = sum + 1;
    }
  }
  cout << sum;
  return 0;
}
