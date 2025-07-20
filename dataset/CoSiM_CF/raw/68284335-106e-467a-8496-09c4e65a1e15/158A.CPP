#include <stdio.h>
int main() {
  int n, k, j, l, i = 0;
  scanf("%d %d", &n, &k);
  int a[n];
  while (i < n) {
    scanf("%d%*c", &a[i]);
    i++;
  }
  j = k, i = k;
  while (j < n) {
    if (a[j] == a[k - 1])
      i++;
    j++;
  }
  l = 0, j = 0;
  while (l < i && a[l] > 0) {
    j++;
    l++;
  }
  printf("%d", j);
  return 0;
}
