#include <stdio.h>
#include <string.h>

int x = 0;

int main() {
  int n;
  scanf("%d", &n);

  for (int i = 1; i <= n; i++) {
    char ara[5];
    scanf("%s", ara);

    if (ara[1] == '+') {
      x = x + 1;
      continue;
    }
    if (ara[1] == '-') {
      x = x - 1;
      continue;
    }
  }

  printf("%d\n", x);
  return 0;
}
