#include <bits/stdc++.h>
using namespace std;

int main() {
  long long k, n, w;
  cin >> k >> n >> w;
  long long cost = k * (w * (w + 1) / 2);
  long long borrow = cost - n;
  if (borrow < 0) {
    cout << "0";
  } else {
    cout << borrow;
  }
}