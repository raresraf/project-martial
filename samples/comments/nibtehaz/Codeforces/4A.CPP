#include <math.h>
#include <stdio.h>
#include <string.h>

int main() {
  int a;

  scanf("%d", &a);

  if ((a > 2) && ((a - 2) % 2 == 0))
    printf("YES\n");
  else
    printf("NO\n");

  return 0;
}
