#include <bits/stdc++.h>
#define ll long long
using namespace std;
int main() {
  ll int n, k, sum = 0;
  cin >> n >> k;
  ll int arr[n];
  for (int i = 0; i < n; i++) {
    cin >> arr[i];
  }
  for (int i = 0; i < n; i++) {
    if (arr[i] > 0 && arr[i] >= arr[k - 1]) {
      ++sum;
      if (arr[i] < 0) {
        break;
      }
    }
  }
  cout << sum;
  return 0;
}
