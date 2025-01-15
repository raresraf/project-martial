/* Date: 10.01.16
Problem: 546A - Soldier and Bananas
*/
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string.h>
#include <string>

using namespace std;

int main() {
  int i, n, t, k, w;
  long long sum = 0;

  cin >> k >> n >> w;
  for (i = 1; i <= w; i++) {
    sum = sum + (i * k);
  }
  if (sum <= n) {
    cout << 0 << "\n";
  } else
    cout << sum - n << "\n";

  return 0;
}
