#include <bits/stdc++.h>

using namespace std;

int main() {
  int t, i, c, s, a = 0, b = 0, d = 0;
  cin >> t;
  while (t--) {
    cin >> i;
    cin >> c;
    cin >> s;
    a = a + i;
    b = b + c;
    d = d + s;
  }
  if (a == 0 && b == 0 && d == 0) {
    cout << "YES";
  } else {
    cout << "NO";
  }
  return 0;
}