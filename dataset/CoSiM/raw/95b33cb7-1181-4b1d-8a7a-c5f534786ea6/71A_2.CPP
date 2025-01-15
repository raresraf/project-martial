#include <iostream>
#include <stdio.h>
#include <string.h>
using namespace std;

int main() {
  int len, n;
  char ch[105];

  scanf("%d", &n);

  for (int i = 0; i < n; i++) {
    scanf("%s", ch);
    len = strlen(ch);

    if (len <= 10)
      printf("%s\n", ch);
    else
      printf("%c%d%c\n", ch[0], len - 2, ch[len - 1]);
  }

  return 0;
}
