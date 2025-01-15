#include <stdio.h>
int main() {
  int n, c[200], f = 0, count = 0, sum = 0, sum1 = 0, temp, i, large, j;
  scanf("%d\n", &n);
  for (i = 0; i < n; i++) {
    scanf("%d", &c[i]);
    sum += c[i];
  }
  for (i = 0; i < n; i++) {
    large = i;
    for (j = i + 1; j < n; j++) {
      if (c[j] > c[large]) {
        large = j;
      }
    }
    temp = c[large];
    c[large] = c[i];
    c[i] = temp;
  }
  for (i = 0; i < n; i++) {
    sum1 += c[i];
    count++;
    if (sum1 >= (sum / 2) + 1)
      break;
  }
  printf("%d", count);

  return 0;
}