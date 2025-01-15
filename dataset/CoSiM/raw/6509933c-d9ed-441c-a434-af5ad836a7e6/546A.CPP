#include <bits/stdc++.h>
using namespace std;

int main() {
  long long k, n, w;
  cin >> k >> n >> w;
  long long sum = 0;

  for (int i = 1; i <= w; i++) {
    sum += i * k;
  }

  if (sum > n)
    cout << sum - n;
  else
    cout << 0;
}
