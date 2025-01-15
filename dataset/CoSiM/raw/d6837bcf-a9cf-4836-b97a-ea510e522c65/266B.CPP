// link : https://codeforces.com/problemset/problem/266/B

#include <bits/stdc++.h>
using namespace std;

int a, b, d;

char c[55];

main() {

  cin >> a >> b;

  for (int i = 1; i <= a; i++)
    cin >> c[i];

  int k = b;
  while (b--) {
    for (int i = 1; i <= a; i++) {

      if (c[i] == 'B' && c[i + 1] == 'G') {

        swap(c[i], c[i + 1]);
        ++i;
      }
    }
  }

  for (int i = 1; i <= a; i++)
    cout << c[i];
}
