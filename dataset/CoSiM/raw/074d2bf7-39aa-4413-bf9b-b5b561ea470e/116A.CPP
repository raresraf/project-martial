#include <bits/stdc++.h>

using namespace std;

int main() {

#ifndef ONLINE_JUDGE
  // for getting input from input.txt
  freopen("input.txt", "r", stdin);
  // for writing output to output.txt
  freopen("output.txt", "w", stdout);
#endif

  int n;
  cin >> n;
  int x, y;
  int sum = 0, max = 0;
  while (n--) {
    cin >> x >> y;
    sum = sum - x;
    sum = sum + y;
    max = max > sum ? max : sum;
  }
  cout << max;

  return 0;
}