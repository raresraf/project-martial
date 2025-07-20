// codeforces soldiers and his fucking bananaa

#include <bits/stdc++.h>
using namespace std;
int main() {

  int k, n, w;
  while (cin >> k >> n >> w) {
    int sum = 0;
    int count = 1;
    while (count <= w) {
      sum += (count * k);
      count++;
    }

    if (sum <= n)
      cout << "0" << endl;
    else
      cout << sum - n << endl;
  }
}
