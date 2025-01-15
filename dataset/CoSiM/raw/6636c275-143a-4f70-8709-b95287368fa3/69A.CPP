#include <stdio.h>

int main(void) {
  int x, y, z;
  int sum_x = 0;
  int sum_y = 0;
  int sum_z = 0;
  int n = 0;

  scanf("%d", &n);

  for (int i = 0; i < n; i++) {
    scanf("%d %d %d", &x, &y, &z);

    sum_x += x;
    sum_y += y;
    sum_z += z;
  }

  if (sum_x == 0 && sum_y == 0 && sum_z == 0) {
    printf("YES");
  } else {
    printf("NO");
  }
  return 0;
}
