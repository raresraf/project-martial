#include <bits/stdc++.h>
using namespace std;

int main() {
  long long k, s, n;
  cin >> k >> s >> n;
  n = (n * (n + 1) * k) / 2;

  if (s > n)
    cout << "0";
  else
    printf("%lld", n - s);
}
