#include "stdio.h"

int main() {
  int n;
  char input[4];
  scanf("%d", &n);

  int x = 0;

  for (int i = 0; i < n; i++) {
    scanf("%s", input);

    if (input[1] == '+') {
      x++;
    } else {
      x--;
    }
  }
  printf("%d", x);

  return 0;
}
