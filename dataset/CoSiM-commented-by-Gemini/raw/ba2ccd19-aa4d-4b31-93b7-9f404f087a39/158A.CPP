// sumanth2303 - 16-11-2020
// problem link - https://codeforces.com/problemset/problem/158/A
// partially solved
#include <iostream>
using namespace std;
#include <bits/stdc++.h>

int main() {
  int n, k;
  cin >> n >> k;
  int a, score;
  vector<int> arr;
  for (int i = 0; i < n; i++) {
    cin >> a;
    arr.push_back(a);
  }
  score = arr[k - 1];
  if (arr[0] == 0) {
    cout << 0 << endl;
    return 0;
  }
  for (int j = k; j < n; j++) {
    if (n == k) {
      cout << k << endl;
      return 0;
    }
    if (score != arr[j] && score != 0) {
      cout << j << endl;
      return 0;
    }
  }
  cout << 0 << endl;
  return 0;
}