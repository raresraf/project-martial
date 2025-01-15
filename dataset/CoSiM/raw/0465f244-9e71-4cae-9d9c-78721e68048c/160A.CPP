#include <algorithm>
#include <stdio.h>
using namespace std;

int main() {
  int n;
  scanf("%d", &n);
  int arr[n], sum = 0, sumaux = 0, cant = 0;
  for (int i = 0; i < n; i++) {
    scanf("%d", &arr[i]);
    sum += arr[i];
  }
  sort(arr, arr + n);
  for (int i = n - 1; i >= 0; i--) {
    sum -= arr[i];
    sumaux += arr[i];
    cant++;
    if (sumaux > sum)
      break;
  }
  printf("%d", cant);

  return 0;
}
