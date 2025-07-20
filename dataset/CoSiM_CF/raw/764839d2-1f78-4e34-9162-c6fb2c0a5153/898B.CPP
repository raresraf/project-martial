#include <bits/stdc++.h>
#include <iostream>
using namespace std;
#define lli long long int

int main() {
  lli c, a, b;
  cin >> c;
  cin >> a;
  cin >> b;
  for (lli i = 0; a * i <= c; i++) {
    int val = c - a * i;
    if (val % b == 0) {
      cout << "YES" << endl;
      cout << i << " " << (val / b) << endl;
      return 0;
    }
  }
  cout << "NO" << endl;
  return 0;
}