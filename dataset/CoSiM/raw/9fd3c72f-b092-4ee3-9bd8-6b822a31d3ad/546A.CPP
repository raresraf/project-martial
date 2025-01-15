#include <iostream>
using namespace std;
int main() {
  int k, n, w;
  cin >> k >> n >> w;
  cout << ((w * (2 * k + (w - 1) * k) / 2) > n
               ? (w * (2 * k + (w - 1) * k) / 2) - n
               : 0)
       << endl;
  return 0;
}