#include <stdio.h>

int main() {
  short x, y, z, n;
  int sumx = 0, sumy = 0, sumz = 0;
  scanf("%hd", &n);
  while (n--) {
    scanf("%hd %hd %hd", &x, &y, &z);
    sumx += x;
    sumy += y;
    sumz += z;
  }
  if (sumx == 0 && sumy == 0 && sumz == 0)
    printf("%s", "YES");
  else
    printf("%s", "NO");

  return 0;
}
