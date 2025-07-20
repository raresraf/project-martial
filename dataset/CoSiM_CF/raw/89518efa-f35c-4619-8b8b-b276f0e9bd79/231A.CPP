#include <algorithm>
#include <cstdio>
#include <iostream>
using namespace std;

int main() {
  int n, a, b, c, res = 0;

  cin >> n;
  for (int i = 0; i < n; ++i) {
    cin >> a >> b >> c;
    if (a + b + c >= 2)
      ++res;
  }

  cout << res << endl;

  return 0;
}
