#include <iostream>

using namespace std;

int main() {
  int n = 0, x = 0, y = 0, z = 0, s = 0;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> x >> y >> z;
    if (x == 1 && y == 1) {
      s++;
    } else if (x == 1 && z == 1) {
      s++;
    } else if (y == 1 && z == 1) {
      s++;
    }
  }
  cout << s << endl;
  return 0;
}
