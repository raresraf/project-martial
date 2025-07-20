#include <stdio.h>

int main(void) {

  //
  //     A 65 ~ Z 90
  //     a 97 ~ z 122

#define ASC_CHNG 32
  char str[2][100];

  for (int i = 0; i < 2; i++) {
    scanf("%s", str[i]);

    for (int j = 0; str[i][j] != '\0'; j++) {
      if (str[i][j] >= 97) {
        str[i][j] = str[i][j] - ASC_CHNG;
      }
    }
  }

  for (int j = 0; str[0][j] != '\0'; j++) {
    if (str[0][j] > str[1][j]) {
      printf("1");
      return 0;
    }
    if (str[0][j] < str[1][j]) {
      printf("-1");
      return 0;
    }
  }

  printf("0");
  return 0;
}
