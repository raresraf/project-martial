#include "stdio.h"
#include "string.h"

int main() {
  char str[101];
  scanf("%s", &str);

  int len = strlen(str);

  if (len < 8) {
    printf("NO");
    return 0;
  }

  int count = 1;

  for (int i = 1; i < len; i++) {
    if (str[i] == str[i - 1]) {
      count++;

      if (count >= 7) {
        printf("YES");
        return 0;
      }
    } else {
      count = 1;
    }
  }

  printf("NO");

  return 0;
}
