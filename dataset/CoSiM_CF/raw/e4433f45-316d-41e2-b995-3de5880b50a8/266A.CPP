#include <stdio.h>
#include <string.h>
int main() {
  char a[50];
  int n, count = 0, i;
  scanf("%d", &n);
  scanf("%s", a);
  for (i = 0; i < strlen(a); i++) {
    if (a[i] == a[i + 1]) {
      count++;
    }
  }
  printf("%d", count);
  return 0;
}