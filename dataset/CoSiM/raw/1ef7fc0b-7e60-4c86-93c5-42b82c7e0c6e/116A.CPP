#include <math.h>
#include <stdio.h>
#include <string.h>

int main() {
  int n, a, b, max = 0, seat = 0;

  scanf("%d", &n);

  while (n--) {
    scanf("%d %d", &a, &b);
    seat -= a;
    seat += b;
    if (seat > max)
      max = seat;
  }

  printf("%d\n", max);

  return 0;
}
