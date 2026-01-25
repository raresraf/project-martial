// https://codeforces.com/problemset/problem/160/A
#include <bits/stdc++.h>
#include <iostream>

using namespace std;

void swapArr(int *xp, int *xy) {
  int temp = *xp;
  *xp = *xy;
  *xy = temp;
}

void sortdesc(int arr[], int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n - 1; j++) {
      if (arr[i] > arr[j]) {
        swapArr(&arr[i], &arr[j]);
      }
    }
  }
}

int main(void) {
  int n;
  cin >> n;

  int coinArr[n];
  int sum = 0;

  for (int i = 0; i < n; i++) {
    cin >> coinArr[i];
    sum += coinArr[i];
  }

  sortdesc(coinArr, n);

  int sumMe = 0;
  int takeCoin = 0;

  for (int i = 0; i < n; i++) {
    if (sumMe <= sum) {
      sumMe += coinArr[i];
      sum -= coinArr[i];
      takeCoin++;
    }
  }
  cout << takeCoin;
}