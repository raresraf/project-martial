#include <stdio.h>
#include <string.h>

int main() {
  int n, j = 0, c, i;
  scanf("%d", &n);
  scanf("%d", &c);
  int a[50];
  for (i = 0; i < n; i++) {
    scanf("%d", &a[i]);
  }
  for (i = 0; i < n; i++) {
    if (a[i] >= a[c - 1] && a[i] > 0) {
      j++;
    }
  }

  printf("%d", j);
}
