#include <iostream>

using namespace std;

long long d[11111];

int main() {
  int n;
  cin >> n;
  int x, y;
  cin >> x >> y;
  d[0] = y;
  long long ans = 0;
  for (int i = 1; i < n; ++i) {
    cin >> x >> y;
    d[i] = d[i - 1] + y - x;
  }
  for (int i = 0; i < n; ++i)
    ans = max(ans, d[i]);
  cout << ans;
}