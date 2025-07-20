#include <stdio.h>
#include <string.h>
int main() {
  char s[100];
  scanf("%s", s);

  int length = strlen(s);

  for (int i = 0; i < length; i++) {
    if (s[i] == 'A' || s[i] == 'E' || s[i] == 'I' || s[i] == 'O' ||
        s[i] == 'U' || s[i] == 'Y' || s[i] == 'a' || s[i] == 'e' ||
        s[i] == 'i' || s[i] == 'o' || s[i] == 'u' || s[i] == 'y')
      continue;
    else {
      printf(".");
      if (s[i] >= 'A' && s[i] <= 'Z') {
        printf("%c", s[i] + 32);
      } else
        printf("%c", s[i]);
    }
  }

  printf("\n");
  return 0;
}
