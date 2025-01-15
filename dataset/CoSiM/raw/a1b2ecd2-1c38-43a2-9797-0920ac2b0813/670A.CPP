#include <stdio.h>
#include <string.h>
int main() {
  long long int x;
  scanf("%lld", &x);
  int sum = 0;
  sum = ((x / 7) * 2);

  if ((x % 7) > 5) {
    printf("%d %d", sum + 1, sum + 2);
  } else {
    if ((x % 7) < 2) {
      printf("%d %d", sum, sum + (x % 7));

    } else {
      printf("%d %d", sum, sum + 2);
    }
  }

  return 0;
}