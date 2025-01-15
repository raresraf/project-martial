#include "stdio.h"

int main() {
  char hello[5] = {'h', 'e', 'l', 'l', 'o'};

  char str[101];

  scanf("%s", str);

  int i = 0;
  int p = 0;
  while (str[i] != '\0') {
    if (str[i] == hello[p]) {
      p++;
    }
    if (p > 4) {
      printf("YES");
      return 0;
    }
    i++;
  }

  printf("NO");
  return 0;
}
