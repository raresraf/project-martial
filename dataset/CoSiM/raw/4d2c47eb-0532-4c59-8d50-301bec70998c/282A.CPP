#include <math.h>
#include <stdio.h>
#include <string.h>

int main() {
  int p = 0, n;
  char inp[4];

  scanf("%d", &n);

  while (n--) {
    scanf("%s", inp);

    if ((!strcmp(inp, "++X")) || (!strcmp(inp, "X++")))
      p++;
    else
      p--;
  }

  printf("%d\n", p);

  return 0;
}
