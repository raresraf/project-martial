#include <stdio.h>
#include <string.h>

int main(void) {
  char arr[100];
  int n, i, j;

  scanf("%d", &n);

  for (i = 1; i <= n; i++) {
    scanf("%s", arr);
    if (strlen(arr) > 10)
      printf("%c%d%c\n", arr[0], strlen(arr) - 2, arr[strlen(arr) - 1]);
    else
      puts(arr);
  }
  return 0;
}
