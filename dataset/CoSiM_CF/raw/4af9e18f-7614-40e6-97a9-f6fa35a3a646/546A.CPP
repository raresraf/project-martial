#include <bits/stdc++.h>
using namespace std;
long long k, n, w, sum;
int main() {
  cin >> k >> n >> w;
  sum = w * (w + 1) / 2;
  sum *= k;
  if (sum <= n)
    cout << 0;
  else
    cout << sum - n;
}