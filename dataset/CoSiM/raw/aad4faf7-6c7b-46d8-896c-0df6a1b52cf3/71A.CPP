#include <stdio.h>
#include <string.h>

int main() {
  int t;
  scanf("%d", &t);
  for (int i = 1; i <= t; i++) {
    char words[101];
    scanf("%s", words);
    if (strlen(words) <= 10)
      printf("%s\n", words);
    else
      printf("%c%d%c\n", words[0], strlen(words) - 2, words[strlen(words) - 1]);
  }
}
