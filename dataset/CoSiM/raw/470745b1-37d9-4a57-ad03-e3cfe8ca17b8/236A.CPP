#include <math.h>
#include <stdio.h>
#include <string.h>

int main() {
  char inp[105];

  while (scanf("%s", inp) != EOF) {
    int i, cnt = 0;

    int alpha[26] = {
        0,
    };

    for (i = 0; inp[i]; i++) {
      if (alpha[inp[i] - 97] == 0) {
        alpha[inp[i] - 97] = 1;
        cnt++;
      }
    }
    if (cnt % 2)
      printf("IGNORE HIM!\n");
    else
      printf("CHAT WITH HER!\n");
  }

  return 0;
}
