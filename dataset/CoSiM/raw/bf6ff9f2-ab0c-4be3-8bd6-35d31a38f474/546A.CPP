#include <iostream>
using namespace std;
main() {
  int k, n, w;
  cin >> k >> n >> w;
  int sum;
  sum = (w * (w + 1) / 2) * k;
  int borrow = sum - n;
  if (borrow < 0)
    cout << "0" << endl;
  else
    cout << borrow << endl;
}
