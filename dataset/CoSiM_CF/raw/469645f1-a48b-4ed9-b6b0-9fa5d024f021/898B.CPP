#include <bits/stdc++.h>
using namespace std;
int main() {
  long long x, y, n, flag = 0, a, b;
  cin >> n >> x >> y;
  for (int i = 0; i <= n; i++) {
    if (x * i <= n && (n - i * x) % y == 0) {
      a = i;
      b = (n - i * x) / y;
      flag = 1;
      break;
    }
  }

  if (flag) {
    cout << "YES\n" << a << " " << b;
  } else
    cout << "NO";
}