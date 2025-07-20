#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(void) {
  char input[102], output[202];
  int i;
  int j = 0;
  scanf("%[^\n]", &input);
  for (i = 0; input[i] != '\0'; i++) {
    if (input[i] == 'a' || input[i] == 'A' || input[i] == 'e' ||
        input[i] == 'E' || input[i] == 'i' || input[i] == 'I' ||
        input[i] == 'o' || input[i] == 'O' || input[i] == 'u' ||
        input[i] == 'U' || input[i] == 'y' || input[i] == 'Y')
      continue;
    else {
      output[j] = '.';
      ++j;
      output[j] = tolower(input[i]);
      ++j;
    }
  }
  output[j] = '\0';
  printf("%s", output);
  return 0;
}
