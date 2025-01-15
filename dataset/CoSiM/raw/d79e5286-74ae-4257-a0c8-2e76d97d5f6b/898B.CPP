//
// Created by ManuelLoaiza on 12/16/17.
//
#include <bits/stdc++.h>

using namespace std;

int main() {

  long long n, a, b;
  cin >> n >> a >> b;

  for (long long i = 0; i <= 10000000; i++) {

    if (i * a > n) {
      cout << "NO" << endl;
      break;
    }

    if ((n - i * a) % b == 0) {
      cout << "YES" << endl;
      long long x = i;
      long long y = (n - i * a) / b;
      cout << x << " " << y << endl;
      break;
    }
  }

  return 0;
}
