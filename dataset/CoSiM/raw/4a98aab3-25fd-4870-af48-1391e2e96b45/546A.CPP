// https://codeforces.com/problemset/problem/546/A
// 546A. Soldier and Bananas
#include <bits/stdc++.h>
using namespace std;
int main() {
  int k, n, w, sum = 0;
  cin >> k >> n >> w;
  for (int i = 1; i <= w; i++) {
    sum = sum + i * k;
  }
  cout << abs(n - sum);
}
