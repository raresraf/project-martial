#include <bits/stdc++.h>
#define lli long long int
using namespace std;

int main() {
  lli k, n, w;
  cin >> k >> n >> w;
  lli sn = ((k * w) * (w + 1)) / 2;
  lli res = sn - n;

  cout << (res >= 0 ? res : 0) << '\n';
}
