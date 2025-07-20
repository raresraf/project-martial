#include <stdio.h>
int main() {
  int a, i, k, c[50], sum = 0;
  scanf("%d%d", &a, &k);
  for (i = 0; i < a; i++) {
    scanf("%d", &c[i]);
  }
  for (i = 0; i < a; i++) {
    if (c[i] >= c[k - 1] && c[i] > 0)
      sum++;
  }

  printf("%d", sum);
}