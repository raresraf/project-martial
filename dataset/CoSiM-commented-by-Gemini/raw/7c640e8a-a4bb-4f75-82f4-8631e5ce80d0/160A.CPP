#include <bits/stdc++.h>
#include <iostream>
using namespace std;

int main(int argc, const char *argv[]) {
  int n, sum = 0;
  cin >> n;
  int res = 0, count = 0;
  int arr[n];

  for (int i = 0; i < n; i++) {
    cin >> arr[i];
    sum += arr[i];
  }

  sort(arr, arr + n, greater<int>());

  while (res <= sum / 2) {
    res += arr[count];
    count++;
  }

  cout << count;
  return 0;
}
