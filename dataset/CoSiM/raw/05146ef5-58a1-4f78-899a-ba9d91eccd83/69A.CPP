#include <iostream>

using namespace std;

int main() {
  int n, x, y, z;
  int sux = 0, suy = 0, suz = 0;
  cin >> n;
  for (int i = 0; i < n; ++i) {
    cin >> x >> y >> z;
    sux += x;
    suy += y;
    suz += z;
  }
  if (sux == 0 && suy == 0 && suz == 0)
    cout << "YES"
         << "\n";
  else
    cout << "NO"
         << "\n";
  return 0;
}