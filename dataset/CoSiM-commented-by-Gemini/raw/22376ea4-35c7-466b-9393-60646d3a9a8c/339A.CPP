#include <stdio.h>
#include <string.h>
int main() {
  char s[101];
  scanf("%s", s);
  int length = strlen(s);
  char ara[101];
  int a = 0;

  for (int i = 49; i <= 51; i++) {
    for (int j = 0; j < length; j++) {
      int x = (int)s[j];
      if (x == i) {
        ara[a] = s[j];
        a++;
      }
    }
  }

  for (int b = 0; b < a - 1; b++) {
    printf("%c+", ara[b]);
  }

  printf("%c\n", ara[a - 1]);
  return 0;
}
