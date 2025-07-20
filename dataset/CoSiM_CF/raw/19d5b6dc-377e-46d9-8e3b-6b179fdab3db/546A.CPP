#include <bits/stdc++.h>
#include <iostream>
using namespace std;
int main() {
  long long k, n, w;
  cin >> k >> n >> w;
  long long req = k * (w * (w + 1) / 2);
  long long borrow = req - n;
  if (borrow <= 0)
    cout << 0;
  else
    cout << borrow;
}
