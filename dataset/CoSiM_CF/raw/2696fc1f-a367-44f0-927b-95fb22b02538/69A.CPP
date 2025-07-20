#include <iostream>

int main() {
  int n;
  std::cin >> n;
  int xsum = 0, ysum = 0, zsum = 0;
  int x, y, z;
  while (n--) {
    std::cin >> x >> y >> z;
    xsum += x;
    ysum += y;
    zsum += z;
  }
  if ((xsum == 0) && (ysum == 0) && (zsum == 0)) {
    std::cout << "YES\n";
  } else {
    std::cout << "NO\n";
  }
}
