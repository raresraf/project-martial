#include <iostream>

using namespace std;

int main() {
  unsigned long k, w, n, cost = 0;

  cin >> k;
  cin >> n;
  cin >> w;

  cost = k * w * (w + 1) / 2;
  if (cost > n)
    cout << (cost - n) << endl;
  else
    cout << 0 << endl;
}
