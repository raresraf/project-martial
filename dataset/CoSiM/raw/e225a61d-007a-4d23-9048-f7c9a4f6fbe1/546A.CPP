//
// Created by Conor on 5/28/2020.
//

#include <iostream>

using namespace std;

int main() {
  int k, n, w;
  cin >> k >> n >> w;

  int cost = 0;

  for (int i = 1; i <= w; ++i) {
    cost += i * k;
  }

  if (cost <= n) {
    cout << 0;
  } else {
    cout << cost - n;
  }

  return 0;
}