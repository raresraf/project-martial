// 160A.cpp
// A. Twins
// Author: Ikki
// Jun 10, 2015

#include <algorithm>
#include <functional>
#include <iostream>
using namespace std;

int main() {
  int n, a[100], sum = 0;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> a[i];
    sum += a[i];
  }
  sort(a, a + n, greater<int>());
  int tmp = 0;
  for (int i = 0; i < n; i++) {
    tmp += a[i];
    if (tmp > sum - tmp) {
      cout << i + 1 << endl;
      break;
    }
  }
  return 0;
  ;
}