#include <stdio.h>
#include <string.h>

// Cannot use strlwr. Dunno why
char *toLower(char *words) {
  for (int i = 0; i < strlen(words); i++) {
    if (words[i] >= 65 && words[i] <= 90)
      words[i] += 32;
  }

  return words;
}

int main() {
  char words1[101];
  char words2[101];

  scanf("%s", words1);
  scanf("%s", words2);

  strcpy(words1, toLower(words1));
  strcpy(words2, toLower(words2));

  printf("%d\n", strcmp(words1, words2));

  return 0;
}
