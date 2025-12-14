#include <iostream>

using namespace std;

int main() {
  int k, n, w, t, b;
  cin >> k >> n >> w;
  t = (k * w * (w + 1)) / 2;
  b = t - n;
  if (b > 0)
    cout << b;
  else
    cout << "0";
  return 0;
}
