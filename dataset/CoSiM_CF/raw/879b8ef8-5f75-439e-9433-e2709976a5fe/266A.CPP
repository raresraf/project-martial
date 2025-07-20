#include <stdio.h>
#include <string.h>
int main() {
  int n, i, count = 0;
  scanf("%d", &n);
  char ch[n], temp;
  scanf("%s", ch);
  for (i = 0; i < n; i++) {
    if (ch[i] != temp || i == 0) {
      temp = ch[i];
    }
    if (temp == ch[i + 1])
      count++;
  }
  printf("%d", count);

  return 0;
}