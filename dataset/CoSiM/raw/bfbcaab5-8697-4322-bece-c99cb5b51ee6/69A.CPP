#include <iostream>
#include <vector>

using namespace std;

int main() {
  int n, x, y, z, sumX = 0, sumY = 0, sumZ = 0;
  std::cin >> n;
  for (int i = 0; i < n; i++) {
    std::cin >> x >> y >> z;
    sumX += x;
    sumY += y;
    sumZ += z;
  }
  if (sumX == 0 && sumY == 0 && sumZ == 0)
    std::cout << "YES";
  else
    std::cout << "NO";
  return 0;
}