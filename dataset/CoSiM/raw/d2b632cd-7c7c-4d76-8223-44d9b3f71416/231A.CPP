#include <iostream>
using namespace std;
int main() {
  int n, x, y, z, count = 0;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> x >> y >> z;
    if ((x == 0 && (y == 1 && z == 1)) || (y == 0 && (x == 1 && z == 1)) ||
        (z == 0 && (y == 1 && x == 1)) || (x == 1 && (y == 1 && z == 1)))
      count += 1;
  }
  cout << count << endl;
  return 0;
}