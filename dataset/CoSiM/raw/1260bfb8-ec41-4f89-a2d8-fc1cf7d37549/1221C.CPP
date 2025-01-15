#include <bits/stdc++.h>
using namespace std;
#define fr(m, n) for (int i = m; i < n; i++)
#define rfr(m, n) for (int i = n; i >= 0; i--)
#define fr1(k, l) for (int j = k; j < l; j++)
#define ll long long int
#define pb push_back
#define set(a) memset(a, -1, sizeof(a))
#define clr(a) memset(a, 0, sizeof(a))
#define e "\n"
typedef vector<int> ve;
typedef vector<bool> vb;

main() {
  // freopen("input.txt","r",stdin);
  int tc, cse = 0;
  cin >> tc;
  while (tc--) {
    int aa, bb, c, a, b;
    cin >> aa >> bb >> c;
    a = max(aa, bb);
    b = min(aa, bb);

    int left = a - b;
    int now_c = left + c;
    if (now_c >= b)
      cout << b << e;
    else {
      int ans = now_c;
      a = a - now_c - left;
      b = b - now_c;
      // cout<<a<<" "<<b<<e;
      cout << ans + (a + b) / 3 << e;
    }
  }
}
