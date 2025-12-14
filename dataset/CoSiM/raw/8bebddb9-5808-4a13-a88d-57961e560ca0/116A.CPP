#include <algorithm>
#include <iostream>
using namespace std;

int main() {
  int n, t, m = 0, s = 0;
  cin >> n;
  while (n--) {
    cin >> t;
    s -= t;
    cin >> t;
    s += t;
    m = max(m, s);
  }
  cout << m;
  return 0;
}