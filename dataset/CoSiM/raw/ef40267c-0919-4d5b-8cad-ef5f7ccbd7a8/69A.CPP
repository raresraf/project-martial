#include <cmath>
#include <cstdio>

int n, x, y, z, sum_x, sum_y, sum_z;
int main() {
  scanf("%d", &n);

  for (int i = 0; i < n; i++) {
    scanf("%d%d%d", &x, &y, &z);

    sum_x += x;
    sum_y += y;
    sum_z += z;
  }

  printf((std::abs(sum_x) + std::abs(sum_y) + std::abs(sum_z)) == 0 ? "YES"
                                                                    : "NO");

  return 0;
}