#include <bits/stdc++.h>
#include <iostream>

using namespace std;

int main() {
  int n;
  cin >> n;
  int A[n], sum = 0, count = 0, sum1 = 0;
  for (int i = 0; i < n; i++) {
    cin >> A[i];
    sum += A[i];
  }

  sort(A, A + n);

  for (int i = n - 1; i >= 0; i--) {
    if (sum1 <= sum / 2) {
      sum1 += A[i];
      count++;
    } else
      break;
  }

  cout << count;

  return 0;
}
