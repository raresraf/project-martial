#include <bits/stdc++.h>

using namespace std;

#define len 60
int arr[len];

int main() {
  int n, k;
  scanf("%d %d", &n, &k);
  int sum = 0;
  int i;
  for (i = 0; i < n; i++) {
    scanf("%d", &arr[i]);
    if (arr[i] > 0 && (i + 1 <= k || arr[i] == arr[k - 1]))
      sum++;
  }

  // if(k + 1 <= n && arr[i] == arr[i - 1]) sum++;
  printf("%d\n", sum);

  return 0;
}
