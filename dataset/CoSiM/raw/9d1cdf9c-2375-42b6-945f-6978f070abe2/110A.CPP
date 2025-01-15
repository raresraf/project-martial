#include <bits/stdc++.h>

using namespace std;

int main() {
  long long int t, i, c = 0, s, a = 0, b = 0;
  cin >> t;
  while (t != 0) {
    i = t % 10;
    if (i == 4 || i == 7) {
      a = a + 1;
    }
    t = t / 10;
  }
  while (a != 0) {
    s = a % 10;
    c = c + 1;
    if (s == 4 || s == 7) {
      b = b + 1;
    }
    a = a / 10;
  }
  if (c == b && c != 0) {
    cout << "YES";
  } else {
    cout << "NO";
  }
  return 0;
}