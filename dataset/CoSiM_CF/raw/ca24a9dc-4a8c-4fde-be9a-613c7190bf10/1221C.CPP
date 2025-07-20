using namespace std;
#include <bits/stdc++.h>
typedef long long int ll;
typedef double dl;
#define pb push_back
#define bg begin()
#define en end()
#define rbg rbegin()
#define ren rend()
#define sz size()
#define r0 return 0
#define F first
#define S second
main() {
  int t;
  cin >> t;
  while (t--) {
    ll c, m, x, ans = 0, bal = 0;
    cin >> c >> m >> x;
    if (c == 0 || m == 0) {
      cout << "0" << endl;
    } else {
      bal = min(c, m);
      ans = (c + m + x) / 3;
      cout << min(ans, bal) << endl;
    }
  }
}
