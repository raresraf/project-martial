#include <stdio.h>
void swap(int *a, int *b) {
  int t = *a;
  *a = *b;
  *b = t;
}
int partition(int arr[], int low, int high) {
  int pivot = arr[high];
  int i = (low - 1);

  for (int j = low; j <= high - 1; j++) {
    if (arr[j] <= pivot) {
      i++;
      swap(&arr[i], &arr[j]);
    }
  }
  swap(&arr[i + 1], &arr[high]);
  return (i + 1);
}
void quickSort(int arr[], int low, int high) {
  if (low < high) {
    int pi = partition(arr, low, high);
    quickSort(arr, low, pi - 1);
    quickSort(arr, pi + 1, high);
  }
}
int main() {
  int n, sum = 0, max = 0;
  scanf("%d", &n);
  int a[n];

  for (int i = 0; i < n; i++)
    scanf("%d", &a[i]);

  quickSort(a, 0, n - 1);

  for (int i = 0; i < n; i++)
    sum = sum + a[i];
  int b = n - 1, count = 0;
  while (1) {
    if (max <= sum) {
      sum = sum - a[b];
      max = max + a[b];
      count++;
      b--;
    } else
      break;
  }
  printf("%d\n", count);
  return 0;
}
