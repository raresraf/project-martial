#include <iostream>
using namespace std;
int main() {

  int k, n, w;
  long long int ossz;
  cin >> k >> n >> w;
  ossz = ((2 * k + (w - 1) * k) * w) / 2;
  ossz - n <= 0 ? cout << "0" : cout << ossz - n;

  return 0;
}
