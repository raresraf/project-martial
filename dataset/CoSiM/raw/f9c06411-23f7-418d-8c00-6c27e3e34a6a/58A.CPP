#include <stdio.h>
#include <string.h>
int main(void) {
  char s[105], h[6] = {"hello"};
  int n, cnt, i, j;
  while (scanf("%s", s) != EOF) {
    j = 0;
    cnt = 0;
    n = strlen(s);
    for (i = 0; i < n; i++) {
      if (s[i] == h[j]) {
        cnt++;
        j++;
      }
    }

    if (cnt == 5)
      printf("YES\n");
    else
      printf("NO\n");
  }

  return 0;
}
