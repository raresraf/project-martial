#include <bits/stdc++.h>

using namespace std;

int main() {

  int k, w, n, sum = 0;

  cin >> k >> n >> w;

  for (int i = 1; i <= w; i++) {

    sum = sum + i * k;
  }

  sum = sum - n;

  if (sum > 0) {

    cout << sum << endl;
  }

  else {
    sum = 0;

    cout << sum << endl;
  }

  return 0;
}
