#include <bits/stdc++.h>
using namespace std;

int main() {
  int k, n, w;
  cin >> k >> n >> w;
  int sum = 0, res = 0;
  for (int i = 1; i <= w; i++) {
    sum += (i * k);
  }
  res = sum > n ? sum - n : 0;
  cout << res;

  return 0;
}
