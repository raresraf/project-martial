#include <bits/stdc++.h>
using namespace std;

int main() {
  long long k, n, w;
  cin >> k >> n >> w;
  long long s = 0;
  for (long long i = 1; i <= w; i++) {
    long long t = i * k;
    s += t;
  }
  if (s > n)
    cout << s - n << endl;
  else
    cout << 0 << endl;

  return 0;
}
