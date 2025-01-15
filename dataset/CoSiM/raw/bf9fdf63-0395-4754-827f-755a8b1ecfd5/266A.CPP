#include "stdio.h"

int main() {
  int len;
  scanf("%d", &len);

  char str[51];
  scanf("%s", &str);

  if (len <= 1) {
    printf("0");
    return 0;
  }

  int rm_count = 0;

  for (int i = 1; i < len; i++) {
    if (str[i] == str[i - 1]) {
      rm_count++;
    }
  }

  printf("%d", rm_count);

  return 0;
}