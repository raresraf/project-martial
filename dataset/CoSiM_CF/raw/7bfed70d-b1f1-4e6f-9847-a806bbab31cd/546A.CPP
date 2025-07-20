#include <bits/stdc++.h>
using namespace std;
int main() {
  // k -> cost of first banana
  // n -> initial no of dollars soldier has
  // w -> no. of bananas he want

  int k, n, w, ans;
  cin >> k >> n >> w;
  w = w * (w + 1) / 2;
  ans = w * k - n;
  if (ans <= 0) {
    ans = 0;
  }
  cout << ans << endl;
  return 0;
}