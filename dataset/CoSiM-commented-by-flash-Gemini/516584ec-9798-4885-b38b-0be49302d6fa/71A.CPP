#include <stdio.h>
#include <string.h>
int main() {
  int n, i, l;
  scanf("%d", &n);
  char ch[100];
  for (i = 0; i < n; i++) {
    scanf("%s", &ch);
    l = strlen(ch);
    if (l <= 10) {
      printf("%s\n", ch);
    } else {
      printf("%c", ch[0]);
      printf("%d", l - 2);
      printf("%c\n", ch[l - 1]);
    }
  }

  return 0;
}