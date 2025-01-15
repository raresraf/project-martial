#include <stdio.h>
int main() {
  int n, k, i, j = 0;
  scanf("%d%d", &n, &k);
  int a[n];
  for (i = 0; i < n; i++)
    scanf("%d", &a[i]);

  if (a[0] > 0) {
    i = 0;
    for (i = 0; (a[i] >= a[k - 1]) && (i < n) && a[i] != 0; i++)
      j++;
  }
  printf("%d", j);
  return 0;
}