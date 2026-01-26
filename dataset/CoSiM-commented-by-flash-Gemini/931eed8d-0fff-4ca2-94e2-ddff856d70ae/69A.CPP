#include <iostream>
using namespace std;

int main() {
  int n, i;
  int x, y, z;
  int xt = 0, yt = 0, zt = 0;
  cin >> n;
  for (i = 0; i < n; i++) {
    cin >> x >> y >> z;
    xt = xt + x;
    yt = yt + y;
    zt = zt + z;
  }
  if (xt == 0 && yt == 0 && zt == 0) {
    cout << "YES";
  } else {
    cout << "NO";
  }
  return 0;
}