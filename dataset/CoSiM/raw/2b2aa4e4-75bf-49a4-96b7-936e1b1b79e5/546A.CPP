#include <bits/stdc++.h>
using namespace std;

typedef long long int lli;

int main() {
  lli w, n, k, i, sum = 0;
  cin >> k >> n >> w;
  for (i = 1; i <= w; i++) {
    sum = sum + i * k;
  }
  if (sum > n)
    sum = sum - n;
  else
    sum = 0;

  cout << sum << endl;

  return 0;
}
