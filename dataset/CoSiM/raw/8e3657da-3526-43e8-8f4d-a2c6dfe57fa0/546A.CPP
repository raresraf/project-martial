#include <bits/stdc++.h>

using namespace std;

int main() {
  long long int cost = 0, n, w, k, a;

  cin >> k >> n >> w;

  for (int i = 1; i <= w; i++) {
    cost = cost + i * k;
  }
  if (cost > n) {
    a = cost - n;
  } else
    a = 0;

  cout << a;
}
