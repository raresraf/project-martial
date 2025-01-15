#include <stdio.h>

//     a 97 ~ z 122

int main(void) {
  char name[100] = {
      0,
  };
  int chk[28] = {
      0,
  };
  int sum = 0;

  scanf("%s", name);

  for (int i = 0; name[i] != '\0'; i++) {
    chk[name[i] - 97] = 1;
  }

  for (int i = 0; i < 28; i++) {
    sum += chk[i];
  }

  if (sum % 2) {
    printf("IGNORE HIM!");
  } else {
    printf("CHAT WITH HER!");
  }

  return 0;
}
