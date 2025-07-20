#include <bits/stdc++.h>
using namespace std;

int main() {

  int n;
  cin >> n;
  int arr[n], bsum = 0, asum = 0;

  for (int i = 0; i < n; i++) {
    cin >> arr[i];
    asum = asum + arr[i];
  }

  int half = asum / 2;

  sort(arr, arr + n);

  for (int i = n - 1; i >= 0; i--) {
    bsum = bsum + arr[i];
    if (bsum > half) {
      int noofcoins = n - i;
      cout << noofcoins << endl;
      break;
    }
  }

  return 0;
}
