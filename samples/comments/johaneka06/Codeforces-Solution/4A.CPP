#include <stdio.h>

int main() {
  int n;
  scanf("%d", &n);
  (n % 2 == 0 && n != 2) ? printf("YES\n") : printf("NO\n");
  return 0;
}
