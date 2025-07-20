#include <bits/stdc++.h>
#include <iostream>
using namespace std;

int main() {
  // your code goes here
  int n, k;
  int count = 0;
  cin >> n >> k;
  int arr[n];
  for (int i = 0; i < n; i++) {
    cin >> arr[i];
  }
  for (int i = 0; i < n; i++) {
    int p = arr[k - 1];

    if (arr[i] >= p && arr[i] > 0) {
      count++;
    }
  }
  cout << count << endl;

  return 0;
}
