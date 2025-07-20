#include <stdio.h>
#include <string.h>
int main() {
  int n;
  scanf("%d", &n);
  char s[100];

  for (int i = 1; i <= n; i++) {
    scanf("%s", s);
    int length = strlen(s);
    if (length <= 10) {
      printf("%s\n", s);
      continue;
    } else {
      printf("%c%d%c\n", s[0], length - 2, s[length - 1]);
    }
  }
  return 0;
}
