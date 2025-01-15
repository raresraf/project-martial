#include <stdio.h>
#include <string.h>

int main() {
  char s[101], t[101];
  scanf("%s", s);
  scanf("%s", t);
  int length1 = strlen(s);

  for (int i = 0; i < length1; i++) {
    if (s[i] >= 'A' && s[i] <= 'Z')
      s[i] = s[i] + 32;

    if (t[i] >= 'A' && t[i] <= 'Z')
      t[i] = t[i] + 32;
  }

  int flag;

  for (int i = 0; i < length1; i++) {
    if (s[i] < t[i]) {
      printf("-1\n");
      flag = 0;
      break;
    }

    else if (s[i] > t[i]) {
      printf("1\n");
      flag = 0;
      break;
    } else {
      flag = 1;
      continue;
    }
  }

  if (flag == 1)
    printf("0\n");

  return 0;
}
