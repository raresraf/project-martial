#include "bits/stdc++.h"
#define do_not_sync                                                            \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(NULL);                                                               \
  cout.tie(NULL);

#define gcd(a, b) __gcd((a), (b))
#define repi(a, n) for (int i = (a); i < (n); i++)
#define repd(n, a) for (int i = (n); i >= (a); i--)

using namespace std;

int main(void) {
  do_not_sync

      // freopen("input.txt", "r", stdin);
      // freopen("output.txt", "w", stdout);

      int n;
  cin >> n;

  vector<int> res;

  for (int i = 1; n >= i; i++) {
    res.emplace_back(i);
    n -= i;
  }

  int s = res.size();
  for (int i = s - 1; i >= 0 && n > 0; i--) {
    res[i] += 1;
    n--;
  }

  cout << s << endl;
  repi(0, s) cout << res[i] << " ";
  cout << endl;

  return 0;
}
