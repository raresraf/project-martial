#include <bits/stdc++.h>
using namespace std;
int main() {
  int k, n, w;
  int total = 0;
  int sol;

  cin >> k >> n >> w;
  for (int i = 1; i <= w; i++) {
    total += k * i;
  }
  if (total < n) {
    cout << "0" << endl;
  } else {
    sol = total - n;
    cout << sol << endl;
  }
  return 0;
}