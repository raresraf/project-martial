// Codeforces
// 546A - Soldier and bananas

#include <cmath>
#include <iostream>
using namespace std;

int main() {
  long long k, n, w;
  cin >> k >> n >> w;
  cout << max((long long)(0), k * w * (w + 1) / 2 - n);

  return 0;
}
