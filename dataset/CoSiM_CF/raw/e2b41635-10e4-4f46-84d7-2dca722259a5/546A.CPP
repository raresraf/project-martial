#include <bits/stdc++.h>

using namespace std;

int main() {
  long long k, w, n;
  long long ans;
  while (cin >> k >> w >> n) {
    ans = 0;

    for (int i = 1; i <= n; i++) {
      ans += i * k;
    }

    if (w < ans)
      cout << ans - w << '\n';
    else
      cout << "0\n";
  }
}
