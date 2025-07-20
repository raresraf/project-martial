#include <iostream>

int main() {
  std::ios::sync_with_stdio(false);
  int n, tmp_x, tmp_y, tmp_z, x = 0, y = 0, z = 0;
  std::cin >> n;
  while (n--) {
    std::cin >> tmp_x >> tmp_y >> tmp_z;
    x += tmp_x;
    y += tmp_y;
    z += tmp_z;
  }
  if (x == 0 && y == 0 && z == 0)
    std::cout << "YES" << std::endl;
  else
    std::cout << "NO" << std::endl;

  return 0;
}