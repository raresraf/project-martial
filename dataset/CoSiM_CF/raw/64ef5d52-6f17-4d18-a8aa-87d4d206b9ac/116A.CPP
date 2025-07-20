#include <bits/stdc++.h>
using namespace std;
int main() {

  int n, sum = 0, maxi = 0, i, a, b;
  cin >> n;
  for (i = 0; i < n; i++) {

    cin >> a >> b;
    sum = sum - a;
    sum = sum + b;

    if (sum > maxi) {
      maxi = sum;
    }
  }

  cout << maxi << endl;

  return 0;
}

// 666Lucifer666
