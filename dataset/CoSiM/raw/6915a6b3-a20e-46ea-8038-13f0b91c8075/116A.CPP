#include <iostream>
using namespace std;

int main() {
  int n, i, a, b, c = 0, m = 0;
  cin >> n;
  for (i = 1; i <= n; i++) {
    cin >> a >> b;
    c = c - a + b;
    if (c > m)
      m = c;
  }
  cout << m;
  return 0;
}
