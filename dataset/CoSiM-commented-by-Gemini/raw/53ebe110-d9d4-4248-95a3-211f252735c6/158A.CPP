#include <bits/stdc++.h>

using namespace std;
int main() {
  int *arr, n, k, count = 0;
  cin >> n >> k;
  arr = new int[n];
  for (int i = 0; i < n; i++) {
    cin >> arr[i];
    if (arr[i] > 0) {
      if (arr[i] >= arr[k - 1]) {
        count++;
      }
    }
  }
  cout << count << endl;
}
