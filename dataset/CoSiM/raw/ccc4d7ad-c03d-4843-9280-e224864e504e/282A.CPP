#include <stdio.h>
int main() {
  int n, count = 0, i, k;
  char a[3];
  scanf("%d", &n);
  k = n;
  for (i = 0; i < n; i++) {
    scanf("%s", a);
    n = k;
    if (a[1] == '+')
      ++count;
    else
      --count;
  }
  printf("%d", count);
  return 0;
}