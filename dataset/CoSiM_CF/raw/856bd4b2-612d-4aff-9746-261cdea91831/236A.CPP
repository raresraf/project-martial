#include <stdio.h>
#include <string.h>

int main() {
  char words[120];
  int count = 0;

  scanf("%s", words);

  for (int i = 0; i < strlen(words); i++) {
    bool flag = false;
    for (int j = 0; j < i; j++) {
      if (words[i] == words[j]) {
        flag = true;
        break;
      }
    }
    if (!flag)
      count++;
  }

  (count % 2 == 0) ? printf("CHAT WITH HER!\n") : printf("IGNORE HIM!\n");

  return 0;
}
