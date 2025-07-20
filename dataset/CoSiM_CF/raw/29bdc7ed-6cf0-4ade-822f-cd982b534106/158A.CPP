#include <iostream>
#include <stdio.h>
using namespace std;
int main() {
  int n, a, x, co = 0, i;
  scanf("%d %d", &n, &a);
  int arr[n];
  for (i = 0; i < n; i++) {
    scanf("%d", &arr[i]);
  }
  x = arr[a - 1];
  for (i = 0; i < n; i++) {
    if (arr[i] >= x && arr[i] > 0) {
      co++;
    }
  }
  printf("%d\n", co);
  return 0;
}
