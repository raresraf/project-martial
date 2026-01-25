#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>
#define sz(a) int((a).size())
#define pb push_back
#define mp make_pair
#define all(c) (c).begin(), (c).end()
#define tr(c, i) for(typeof((c).begin() i = (c).begin(); i != (c).end(); i++)
#define present(c, x) ((c).find(x) != (c).end())
#define cpresent(c, x) (find(all(c), x) != (c).end())

using namespace std;

void solve() {
  int m, n, ans;
  cin >> m >> n;
  if (n % 2 == 0 || m % 2 == 0) {
    ans = (m * n) / 2;
  } else {
    ans = max(m * (n - 1) / 2 + m / 2, n * (m - 1) / 2 + n / 2);
  }
  cout << ans << "\n";
}

int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(0);
  cout.tie(0);
  solve();
  return 0;
}