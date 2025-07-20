#include "stdio.h"
#include "string.h"

int main() {
  char str[101];

  scanf("%s", &str);

  int len = strlen(str);

  for (int i = 0; i < len - 2; i++) {
    for (int j = 0; j < len - i - 2; j += 2) {
      if (str[j] > str[j + 2]) {
        char temp = str[j];
        str[j] = str[j + 2];
        str[j + 2] = temp;
      }
    }
  }

  printf("%s", str);

  return 0;
}
