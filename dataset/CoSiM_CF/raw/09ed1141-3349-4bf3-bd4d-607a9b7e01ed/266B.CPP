#include <bits/stdc++.h>

using namespace std;
int main() {
  int n, t, i, j;
  scanf("%d %d", &n, &t);
  getchar();
  char arr[n + 1], temp;
  scanf("%s", &arr);
  for (i = 0; i < t; i++) {
    for (j = 0; j < n - 1; j++) {
      if (arr[j] == 'B' && arr[j + 1] == 'G') {
        temp = arr[j];
        arr[j] = arr[j + 1];
        arr[j + 1] = temp;
        j++;
      }
    }
  }
  printf("%s\n", arr);
}
