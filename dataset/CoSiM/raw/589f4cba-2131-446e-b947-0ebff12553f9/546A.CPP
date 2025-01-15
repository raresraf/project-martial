#include <iostream>

using namespace std;

int main() {
  int k, n, w;
  cin >> k >> n >> w;
  int total = 0;
  for (int i = 1; i <= w; i++) {
    total += (i * k);
  }
  int borrow;
  borrow = total - n;
  (borrow > 0) ? cout << borrow : cout << 0;
}
