#include <bits/stdc++.h>
using namespace std;
int main() {
  int n;
  cin >> n;
  int *arr = new int[n];
  for (int i = 0; i < n; i++) {
    int c;
    cin >> c;
    arr[i] = c;
  }
  sort(arr, arr + n);

  int sum1 = 0, sum2, count = 0, j, i;
  for (i = n - 1; i >= 0; i--) {
    int maxval = arr[i];
    sum1 = sum1 + maxval;
    sum2 = 0;
    for (j = i - 1; j >= 0; j--) {
      sum2 = sum2 + arr[j];
    }
    if (sum1 > sum2) {
      count++;
      break;
    } else if (sum1 <= sum2) {
      count++;
    }
  }
  cout << count << "\n";
}
