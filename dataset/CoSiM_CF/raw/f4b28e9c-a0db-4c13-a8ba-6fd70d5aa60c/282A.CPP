#include <stdio.h>
#include <string.h>

int main() {
  int x = 0;
  int t;
  char opr[5];

  scanf("%d", &t);
  for (int i = 0; i < t; i++) {
    scanf("%s", opr);
    if (strstr(opr, "++"))
      x++;
    else
      x--;
  }
  printf("%d\n", x);

  return 0;
}
