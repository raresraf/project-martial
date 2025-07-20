#include <bits/stdc++.h>

typedef long long ll;

using namespace std;

int main() {
  int k, n, w;
  cin >> k >> n >> w;

  ll ans = k * w * (1 + w) / 2 - n;

  if (ans > 0)
    cout << ans << "\n";
  else {
    cout << 0 << "\n";
  }
}

// g++ -std=c++11 -O2 -Wall 546A.cpp -o out