#include <bits/stdc++.h>

using namespace std;

int main() {

  int n, k;
  cin >> n >> k;
  int arr[n];
  int count = 0;

  for (int i = 0; i < n; i++) {
    cin >> arr[i];
  }

  for (int i = 0; i < n; i++) {
    if (arr[i] >= arr[k - 1] and arr[i] > 0) {
      count++;
    }
  }
  cout << count;
}
