#include <iostream>
using namespace std;
int main() {
  int n;
  cin >> n;
  int xsum = 0, ysum = 0, zsum = 0;
  while (n--) {
    int x, y, z;
    cin >> x >> y >> z;
    xsum += x;
    ysum += y;
    zsum += z;
  }
  if (xsum == 0 && ysum == 0 && zsum == 0) {
    cout << "YES" << '\n';
  } else
    cout << "NO" << '\n';
}
