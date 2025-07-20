#include <stdio.h>
#define MAX 255

int main() {
  int t, n;
  int array[255];
  int digit;

  scanf("%d %d", &t, &n);
  for (int i = 0; i < t; i++) {
    scanf("%d", &array[i]);
    if (i + 1 == n)
      digit = array[i];
  }

  int count = 0;
  for (int i = 0; i < t; i++) {
    if (array[i] >= digit && array[i] != 0)
      count++;
  }

  printf("%d\n", count);
  return 0;
}
