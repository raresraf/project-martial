#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, c = 0, x, s, i;
  cin >> n;

  while (n--) {
    for (i = 0, s = 0; i < 3; i++) {
      cin >> x;
      if (x)
        s++;
    }
    if (s > 1)
      c++;
  }
  cout << c;
}
