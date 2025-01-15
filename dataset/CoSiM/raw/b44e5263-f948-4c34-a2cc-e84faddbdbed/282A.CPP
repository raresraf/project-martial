#include <stdio.h>
#include <string.h>

int main() {
  int t = 0, i, n;
  char ch[5];

  scanf("%d", &n);
  for (i = 0; i < n; i++) {
    scanf("%s", ch);
    if (!strcmp(ch, "X++"))
      t++;
    else if (!strcmp(ch, "++X"))
      ++t;
    else if (!strcmp(ch, "X--"))
      t--;
    else if (!strcmp(ch, "--X"))
      --t;
  }
  printf("%d\n", t);

  return 0;
}
