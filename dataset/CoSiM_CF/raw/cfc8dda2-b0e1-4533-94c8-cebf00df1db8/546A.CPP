#include <bits/stdc++.h>
using namespace std;

int main() {
  long int k, n, w, total_cost = 0;

  cin >> k >> n >> w;

  for (long int i = 1; i <= w; i++) {
    total_cost += (k * i);
  }

  cout << total_cost - n << endl;

  return 0;
}

/**
3 17 4
output:
13
**/
