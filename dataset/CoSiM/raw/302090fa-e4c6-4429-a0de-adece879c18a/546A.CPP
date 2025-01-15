#include <stdio.h>

int main(void) {
  int k, n, w;
  int sum = 0;
  scanf("%d %d %d", &k, &n, &w);

  for (int i = 1; i <= w; i++) {
    sum += k * i;
  }
  if (sum - n > 0) {
    printf("%d", sum - n);
  } else {
    printf("0");
  }
  return 0;
}
