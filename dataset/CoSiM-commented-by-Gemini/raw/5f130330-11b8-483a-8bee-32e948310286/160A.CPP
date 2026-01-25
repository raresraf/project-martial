#include <stdio.h>
#include <string.h>
int main() {
  int n, a[100], i, j, S = 0, k, count = 1;
  scanf("%d", &n);
  for (i = 0; i < n; i++) {
    scanf("%d", &a[i]);
    S += a[i];
  }
  for (i = 0; i < n; i++) {
    for (j = 0; j < n - 1; j++)
      if (a[j] > a[j + 1]) {
        a[j] = a[j] + a[j + 1];
        a[j + 1] = a[j] - a[j + 1];
        a[j] = a[j] - a[j + 1];
      }
  }

  for (i = 0; i < n; i++) {
    k += a[n - i - 1];
    if (k <= S - k)
      count++;
    else
      break;
  }
  printf("%d", count);
  return 0;
}