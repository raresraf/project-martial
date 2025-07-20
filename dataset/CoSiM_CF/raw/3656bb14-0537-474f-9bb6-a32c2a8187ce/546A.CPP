#include <bits/stdc++.h>
#include <string>
#define ll long long
using namespace std;

int main() {

#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif
  int n, k, w, sum = 0, total = 0;
  cin >> k >> n >> w;
  for (int i = 1; i <= w; i++) {
    sum += (i * k);
  }
  if (sum > n) {
    total = abs(n - sum);
  }
  cout << total;
  return 0;
}
