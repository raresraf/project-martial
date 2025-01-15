#include <iostream>

using namespace std;

int main() {
  int k, n, w;
  int total = 0, borrow = 0;

  cin >> k >> n >> w;

  for (int i = 1; i <= w; ++i) {
    total += k * i;
  }

  if (n >= total) {
    cout << 0 << endl;
  } else {
    borrow = total - n;
    cout << borrow << endl;
  }

  return 0;
}