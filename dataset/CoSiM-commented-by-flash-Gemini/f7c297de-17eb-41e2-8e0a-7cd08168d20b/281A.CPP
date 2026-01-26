#include <stdio.h>

int main(void) {

  //
  //     A 65 ~ Z 90
  //     a 97 ~ z 122
  //

#define ASC_CHNG 32

  char input[1000];

  scanf("%s", input);

  if (input[0] >= 97) {
    input[0] = input[0] - ASC_CHNG;
  }

  printf("%s", input);

  return 0;
}
