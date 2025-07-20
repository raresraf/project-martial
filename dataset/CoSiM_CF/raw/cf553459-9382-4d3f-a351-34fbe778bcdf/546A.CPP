#include <iostream>
using namespace std;
int main() {
  int k, n, w;
  cin >> k >> n >> w;
  int borrow = (1 + w) * k * w / 2 - n;
  cout << ((borrow > 0) ? borrow : 0);
  return 0;
}
