#include <bits/stdc++.h>

using namespace std;

int main() {

#ifndef ONLINE_JUDGE
  // for getting input from input.txt
  freopen("input.txt", "r", stdin);
  // for writing output to output.txt
  freopen("output.txt", "w", stdout);
#endif

  long long int k, n, w, sum;
  cin >> k >> n >> w;
  sum = (k * (w + 1) * (w)) / 2;

  if (sum - n >= 0) {
    cout << sum - n;
  } else {
    cout << "0";
  }

  return 0;
}