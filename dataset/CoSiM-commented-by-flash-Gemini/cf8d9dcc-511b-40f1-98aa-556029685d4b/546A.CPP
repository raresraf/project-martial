#include <bits/stdc++.h>
using namespace std;
int main() {
  int k, n, w, t;
  cin >> k >> n >> w;
  int borrow;
  int total = 0;
  for (int i = 1; i <= w; i++) {
    t = k * i;
    total = total + t;
  }
  // cout<<total;
  if (total > n) {
    borrow = total - n;
    cout << borrow;
  } else {
    cout << "0";
  }
}
