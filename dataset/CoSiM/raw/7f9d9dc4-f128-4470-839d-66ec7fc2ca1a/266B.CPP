#include <stdio.h>
#include <string.h>
int main() {
  char arr[500];
  int n, t, i, j;
  scanf("%d %d\n", &n, &t);
  scanf("%s", arr);
  for (i = 0; i < t; i++) {
    for (j = 0; j < strlen(arr); j++) {
      if (arr[j] == 'B' && arr[j + 1] == 'G') {
        arr[j] = 'G';
        arr[j + 1] = 'B';
        j++;
      }
    }
  }
  printf("%s", arr);
  return 0;
}