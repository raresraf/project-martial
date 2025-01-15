#include "stdio.h"
#include "string.h"

int main() {
  int n;
  scanf("%d", &n);

  char input[101];

  for (int i = 0; i < n; i++) {
    scanf("%s", input);
    int length = strlen(input);

    char output[6];

    if (length > 10) {
      printf("%c%d%c\n", input[0], length - 2, input[length - 1]);
    } else {
      printf("%s\n", input);
    }
  }

  return 0;
}