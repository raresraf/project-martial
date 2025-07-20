#include <stdio.h>
#include <string.h>

int main(void) {

  int n;
  int x = 0;

  char bit[150][3];

  scanf("%d", &n);

  for (int i = 0; i < n; i++) {
    scanf("%s", bit[i]);

    if (strstr(bit[i], "+") != 0) { //    ++
      x++;
    } else {
      x--;
    }
  }

  printf("%d", x);

  return 0;
}
