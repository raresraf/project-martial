#include <stdio.h>

int main() {
  char words[1001];

  scanf("%s", words);
  if (words[0] > 90)
    words[0] = words[0] - 32;

  printf("%s\n", words);

  return 0;
}
