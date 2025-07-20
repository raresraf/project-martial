#include <math.h>
#include <stdio.h>
#include <string.h>

int main() {
  int t, n;
  char word[105];

  scanf("%d", &t);
  while (t--) {
    scanf("%s", word);
    n = strlen(word);

    if (n <= 10)
      printf("%s\n", word);
    else
      printf("%c%d%c\n", word[0], n - 2, word[n - 1]);
  }

  return 0;
}
