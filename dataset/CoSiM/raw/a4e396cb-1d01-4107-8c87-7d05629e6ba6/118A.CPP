#include <math.h>
#include <stdio.h>
#include <string.h>

int main() {
  int n, i;
  char inp[105];

  scanf("%s", inp);
  n = strlen(inp);

  for (i = 0; i < n; i++) {
    if ((inp[i] == 'A') || (inp[i] == 'a') || (inp[i] == 'E') ||
        (inp[i] == 'e') || (inp[i] == 'I') || (inp[i] == 'i') ||
        (inp[i] == 'O') || (inp[i] == 'o') || (inp[i] == 'U') ||
        (inp[i] == 'u') || (inp[i] == 'Y') || (inp[i] == 'y'))
      continue;
    else if (inp[i] > 'a')
      printf(".%c", inp[i]);

    else
      printf(".%c", inp[i] - 'A' + 'a');
  }
  printf("\n");

  return 0;
}
