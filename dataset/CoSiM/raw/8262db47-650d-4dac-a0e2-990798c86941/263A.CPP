#include <iostream>
#include <stdio.h>
using namespace std;

int main() {
  int arr[5][5], i, j, a, b;
  for (i = 0; i < 5; i++) {
    for (j = 0; j < 5; j++) {
      scanf("%d", &arr[i][j]);
    }
  }
  for (i = 0; i < 5; i++) {
    for (j = 0; j < 5; j++) {
      if (arr[i][j] == 1) {
        a = i;
        b = j;
      }
    }
  }
  a = a - 2;
  b = b - 2;
  if (a < 0) {
    a = a - (2 * a);
  }
  if (b < 0) {
    b = b - (2 * b);
  }
  printf("%d\n", a + b);
  return 0;
}
