#include <bits/stdc++.h>
#include <iostream>
using namespace std;
int main() {
  int i, j, p, q, n;
  for (i = 1; i <= 5; i++) {
    for (j = 1; j <= 5; j++) {
      cin >> n;
      if (n == 1) {
        p = i;
        q = j;
      }
    }
  }
  if (p > 3)
    p = p - 3;
  else
    p = 3 - p;
  if (q > 3)
    q = q - 3;
  else
    q = 3 - q;
  p = p + q;
  cout << p;
  return 0;
}
