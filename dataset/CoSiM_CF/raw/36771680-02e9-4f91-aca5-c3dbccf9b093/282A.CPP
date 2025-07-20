#include <bits/stdc++.h>
using namespace std;
int main() {
  string ch;
  int a, i, x;
  while (cin >> a) {
    x = 0;
    for (i = 1; i <= a; i++) {
      cin >> ch;
      if (ch == "++X" || ch == "X++") {
        x++;
      } else if (ch == "--X" || ch == "X--") {
        x--;
      }
    }
    cout << x << endl;
  }
  return 0;
}
