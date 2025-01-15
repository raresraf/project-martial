#include <iostream>
using namespace std;
int main() {
  int n, x, y, z;
  cin >> n;
  int xsum = 0, ysum = 0, zsum = 0;
  while (n--) {
    cin >> x >> y >> z;
    xsum += x;
    ysum += y;
    zsum += z;
  }
  if (xsum == 0 and ysum == 0 and zsum == 0) {
    cout << "YES";
  } else
    cout << "NO";
}
