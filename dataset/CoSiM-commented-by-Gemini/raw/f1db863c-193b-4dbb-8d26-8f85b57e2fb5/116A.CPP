#include <algorithm>
#include <bits/stdc++.h>
#include <cmath>
#include <iostream>
using namespace std;
int a[100010];
// int b[100010];
int main() {
  int n;
  cin >> n;
  int a, b;
  long int ma = 0, sum = 0;
  for (int i = 0; i < n; i++) {
    cin >> a >> b;
    sum = b - a + sum;
    if (sum > ma)
      ma = sum;
  }
  cout << ma;
  return 0;
}