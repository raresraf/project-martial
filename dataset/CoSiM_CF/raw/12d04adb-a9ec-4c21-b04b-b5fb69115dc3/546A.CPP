#include <bits/stdc++.h>
using namespace std;

int main() {
  long long int k, n, w, w1;
  cin >> k >> n >> w;
  w1 = (k * w * w + k * w) / 2;
  if (w1 > n) {
    cout << w1 - n << endl;
  } else
    cout << "0" << endl;
}
