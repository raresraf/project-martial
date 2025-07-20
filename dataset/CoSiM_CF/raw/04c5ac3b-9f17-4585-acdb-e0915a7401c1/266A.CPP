/**
 * I'm sorry I was an Idiot
 */
#include <stdio.h>

int main() {
  int big;
  scanf("%d", &big);
  char word[big + 1];
  scanf("%s", word);
  char last = word[0];
  int changes = 0;
  for (int i = 1; i < big; ++i) {
    if (word[i] == last)
      changes++;
    else
      last = word[i];
  }
  printf("%d\n", changes);
  return 0;
}