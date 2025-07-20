#include <bits/stdc++.h>

using namespace std;

int main() {

  int a, b, c, n, p = 0;
  cin >> n;

  while (n--) {
    cin >> a >> b >> c;

    p += a + b + c > 1 ? 1 : 0;
  }
  cout << p;
}