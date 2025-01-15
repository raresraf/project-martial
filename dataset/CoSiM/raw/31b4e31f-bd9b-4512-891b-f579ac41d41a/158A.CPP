#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, k, count = 0;
  int arr[105];
  cin >> n >> k;
  for (int i = 1; i <= n; i++) {
    cin >> arr[i];
  }
  for (int i = 1; i <= n; i++) {
    if (arr[k] <= arr[i] && arr[i] > 0) {
      count++;
    }
  }
  cout << count << endl;
  return 0;
}
