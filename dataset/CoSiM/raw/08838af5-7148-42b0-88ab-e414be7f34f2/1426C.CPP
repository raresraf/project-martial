#include <cmath>
#include <iostream>

using namespace std;

int main() {
  int x, n;
  cin >> x;
  while (x) {
    cin >> n;
    int root = floor(sqrt(n));
    if (root * root >= n)
      cout << 2 * (root - 1) << endl;
    else if ((root + 1) * (root) >= n)
      cout << 2 * root - 1 << endl;
    else
      cout << 2 * root << endl;
    --x;
  }

  return 0;
}
