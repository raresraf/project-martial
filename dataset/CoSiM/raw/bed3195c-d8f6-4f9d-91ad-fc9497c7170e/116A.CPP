#include <stdio.h>

int main() {
  int arr[1001][5], t;
  int sum = 0;
  int capacity = 0;
  scanf("%d", &t);

  for (int i = 0; i < t; i++) {
    scanf("%d %d", &arr[i][0], &arr[i][1]);
    if (i == 0) {
      capacity = arr[i][1];
      sum = arr[i][1];
    } else {
      sum = sum - arr[i][0] + arr[i][1];
      if (sum > capacity)
        capacity = sum;
    }
  }

  printf("%d\n", capacity);

  return 0;
}
