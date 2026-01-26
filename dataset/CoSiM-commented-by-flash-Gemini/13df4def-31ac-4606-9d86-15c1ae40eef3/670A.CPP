#include <iostream>
using namespace std;
int main() {
  long n, d, r;
  cin >> n;
  d = n / 7 * 2;
  r = n % 7;
  cout << d + (r == 6) << ' ' << d + (r > 2 ? 2 : r);
}
