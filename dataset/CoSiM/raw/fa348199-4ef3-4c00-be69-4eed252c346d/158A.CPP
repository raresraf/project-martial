/*
Author : hardRocK
Date :
Online Judge : Codeforces
Problem Name : Next Round
Problem Link : http://codeforces.com/problemset/problem/158/A
*/

#include <bits/stdc++.h>
using namespace std;

int main() {

  int n, k, val, cnt = 0;
  cin >> n >> k;

  int arr[n];

  for (int i = 0; i < n; ++i) {
    cin >> arr[i];
  }

  val = arr[k - 1];

  for (int i = 0; i < n; ++i) {
    if (arr[i] >= val && arr[i] >= 1) {
      cnt++;
    }
  }

  cout << cnt << endl;

  return 0;
}
