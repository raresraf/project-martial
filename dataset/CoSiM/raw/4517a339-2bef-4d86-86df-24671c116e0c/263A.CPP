// link : https://codeforces.com/problemset/problem/263/A

#include <bits/stdc++.h>
using namespace std;

int a, b, c, d;

main() {

  for (int i = 1; i <= 5; i++) {

    for (int j = 1; j <= 5; j++) {

      cin >> a;
      if (a == 1) {
        c = i;
        d = j;
      }
    }
  }

  int k = 0;
  k = abs(3 - d) + abs(3 - c);

  cout << k << endl;
}
