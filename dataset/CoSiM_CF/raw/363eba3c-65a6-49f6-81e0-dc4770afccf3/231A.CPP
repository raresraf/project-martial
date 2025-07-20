#include <bits/stdc++.h>
using namespace std;

int main(void) {
  int problems, a, b, c, forSure = 0;

  cin >> problems;

  while (problems--) {
    cin >> a >> b >> c;
    if (a) {
      if (b || c) {
        forSure++;
      }
    } else if (b) {
      if (c) {
        forSure++;
      }
    }
  }

  cout << forSure;

  return 0;
}
