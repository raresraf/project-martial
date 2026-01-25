#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
typedef vector<int> vi;
typedef pair<int, int> pii;
#define pb push_back
#define mp make_pair
#define ff first
#define ss second
#define sz(a) (int)a.size()
#define all(c) c.begin(), c.end()
#define rep(i, n) for (int i = 0; i < (n); i++)
#define rept(i, a, b) for (int i = (a); i < (b); i++)
#define fill(a, v) memset(a, v, sizeof(a))
#define foreach(c, it)                                                         \
  for (__typeof(c.begin()) it = c.begin(); it != c.end(); ++it)

int main() {
  int n;
  cin >> n;
  int a[n + 2][n + 2];
  fill(a, 0);
  char x;
  rep(i, n) {
    rep(j, n) {
      cin >> x;
      if (x == 'x') {
        a[i + 1][j + 1] = 1;
      } else
        a[i + 1][j + 1] = -1;
    }
  }
  int flag = 1;
  for (int i = 1; i <= n; i++) {
    for (int j = 1; j <= n; j++) {
      int count = 0;
      if (a[i][j + 1] == -1)
        count++;
      if (a[i][j - 1] == -1)
        count++;
      if (a[i + 1][j] == -1)
        count++;
      if (a[i - 1][j] == -1)
        count++;

      if (count % 2 == 0)
        continue;
      else {
        flag = 0;
        cout << "NO\n";
        break;
      }
    }
    if (!flag)
      break;
  }
  if (flag)
    cout << "YES\n";

  return 0;
}
