#include <bits/stdc++.h>
using namespace std;

int main() {
  int k, n, w, i, j, sum = 0, a;
  cin >> k >> n >> w;

  for (i = 1; i <= w; i++) {
    sum += (k * i);
  }

  if (sum > n)
    a = sum - n;
  else {
    cout << "0" << endl;
    return 0;
  }

  cout << a << endl;
}
