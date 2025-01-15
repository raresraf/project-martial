#include <iostream>

using std::cin;
using std::cout;
using std::endl;

int main() {
  int n;
  cin >> n;
  int x = 0, y = 0, z = 0;
  int xt = 0, yt = 0, zt = 0;
  for (int i = 0; i != n; ++i) {
    cin >> x >> y >> z;
    xt += x;
    yt += y;
    zt += z;
  }

  if (!xt && !yt && !zt) {
    cout << "YES";
  } else {
    cout << "NO";
  }
  return 0;
}
