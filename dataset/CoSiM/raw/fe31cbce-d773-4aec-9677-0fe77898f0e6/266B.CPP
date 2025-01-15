#include <malloc.h>
#include <stdio.h>

int main(void) {

  int n, t;
  char *s;

  scanf("%d %d", &n, &t);
  s = (char *)malloc(n * sizeof(char)); //

  scanf("%s", s);

  for (int sec = 0; sec < t; sec++) {
    for (int i = 0; s[i + 1] != '\0'; i++) {
      if (s[i] == 'B' && s[i + 1] == 'G') {
        s[i] = 'G';
        s[i + 1] = 'B';
        i++;
      }
    }
  }

  for (int idx = 0; s[idx] != '\0'; idx++) {
    printf("%c", s[idx]);
  }

  return 0;
}
