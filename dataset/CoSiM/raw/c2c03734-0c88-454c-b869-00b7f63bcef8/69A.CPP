#include <bits/stdc++.h>
#include <iostream>
using namespace std;
int main() {
  int n;
  cin >> n;
  int p = 0, q = 0, r = 0;
  int a = 0, b = 0, c = 0;
  for (int i = 0; i < n; i++) {
    cin >> p >> q >> r;
    a += p;
    b += q;
    c += r;
  }
  if ((a == 0) && (b == 0) && (c == 0))
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
  return 0;
}