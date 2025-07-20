#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, x, y, a = 0, c = 0;
  cin >> n;
  while (n--) {
    cin >> x >> y;
    a += y - x;
    if (a > c)
      c = a;
  }
  cout << c;
}
