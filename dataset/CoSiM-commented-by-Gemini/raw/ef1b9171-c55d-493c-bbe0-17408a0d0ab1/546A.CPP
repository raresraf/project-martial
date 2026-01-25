#include <iostream>

using namespace std;

int main() {
  int k, n, w;
  cin >> k >> n >> w;
  if (n >= (k * w * (w + 1)) / 2)
    cout << 0;
  else
    cout << (k * w * (w + 1) / 2) - n;
}