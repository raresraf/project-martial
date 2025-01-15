#include <stdio.h>
#include <string.h>

int main(void) {

  char player[100];

  scanf("%s", player);

  if (strstr(player, "1111111") != 0 || strstr(player, "0000000") != 0) { // 7
    printf("YES");
  } else {
    printf("NO");
  }

  return 0;
}
