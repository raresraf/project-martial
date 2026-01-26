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
  string a, b;
  cin >> a >> b;
  int chara[30] = {0};
  int charb[30] = {0};

  if (a.length() != b.length()) {
    cout << "NO\n";
    return 0;
  } else {
    rep(i, a.length()) { chara[a[i] - 'a']++; }
    rep(i, b.length()) { charb[b[i] - 'a']++; }
    int flag = 1;
    rep(i, 26) {
      if (chara[i] != charb[i])
        flag = 0;
    }

    int count = 0;
    if (flag == 0) {
      cout << "NO\n";
      return 0;
    } else {
      rep(i, a.length()) {
        if (a[i] != b[i])
          count++;
      }

      if (count == 2) {
        cout << "YES\n";
      } else
        cout << "NO\n";
    }
  }

  return 0;
}