#include <stdio.h>
int main() {
  int n, i, a, j, sum = 0, total = 0;
  scanf("%d", &n);
  int arr[n];
  for (i = 0; i < n; i++) {
    scanf("%d", &arr[i]);
    sum = sum + arr[i];
  }
  for (i = 0; i < n; i++) {
    for (j = i + 1; j < n; j++) {
      if (arr[i] < arr[j]) {
        a = arr[i];
        arr[i] = arr[j];
        arr[j] = a;
      }
    }
  }
  for (i = 0; i < n; i++) {
    total = total + arr[i];
    if (total >= ((sum / 2) + 1)) {
      i++;

      break;
    }
  }
  printf("%d", i);
  return 0;
}
