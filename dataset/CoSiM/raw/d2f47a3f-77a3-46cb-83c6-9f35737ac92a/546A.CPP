#include <iostream>
using namespace std;

int main() {
  int k, n, w, a;
  cin >> k >> n >> w;
  a = k * w * (w + 1) / 2 - n;
  if (a > 0)
    cout << a;
  else
    cout << 0;
  return 0;
}
