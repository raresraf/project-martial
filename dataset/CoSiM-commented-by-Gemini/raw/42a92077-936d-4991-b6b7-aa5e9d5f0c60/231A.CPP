#include <bits/stdc++.h>
using namespace std;
int main() {
  ios::sync_with_stdio(false);
  cin.tie(0);

  int n, sum(0), a, b, c;
  cin >> n;

  while (n--) {
    cin >> a >> b >> c;

    if (a + b + c >= 2)
      sum++;
  }
  cout << sum << "\n";
}
