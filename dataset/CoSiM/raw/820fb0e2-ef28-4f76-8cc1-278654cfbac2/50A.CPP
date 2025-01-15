#include <cmath>
#include <iostream>

using namespace std;

int main() {
  double m = 0, n = 0, maxDom = 0, domSize = 2, reqSize = 0, maxD = 0;
  cin >> m >> n;
  reqSize = m * n;
  maxD = reqSize / domSize;
  maxDom = floor(maxD);
  cout << (int)maxDom << endl;
  return 0;
}
