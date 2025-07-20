//============================================================================
// Name        : 160A.cpp
// Author      :
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <bits/stdc++.h>
#include <iostream>

using namespace std;

int main() {
  int n;
  cin >> n;
  int arr[n];
  int sum = 0;
  for (int i = 0; i < n; i++) {
    cin >> arr[i];
    sum += arr[i];
  }
  sort(arr, arr + n, greater<int>());
  int res = 0, sum2 = 0;
  for (int i = 0; i < n; i++) {
    sum2 += arr[i];
    if (sum2 > sum / 2) {
      res++;
      break;
    } else
      res++;
  }
  cout << res;
  return 0;
}
