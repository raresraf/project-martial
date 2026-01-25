// 116A - Tram

#include <iostream>

using namespace std;

int main() {
  int n, a, b, i, mx, sum;

  cin >> n;

  for (i = 0; i < n; i++) {
    cin >> a >> b;

    if (i == 0) {
      sum = b;
      mx = b;
      continue;
    }

    sum -= a;
    sum += b;
    mx = max(mx, sum);
  }

  cout << mx;

  return 0;
}
