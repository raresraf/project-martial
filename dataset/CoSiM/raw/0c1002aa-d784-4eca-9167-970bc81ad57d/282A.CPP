#include <bits/stdc++.h>
using namespace std;
#define debug(x, y) cout << x << y
#define pb push_back
#define mp make_pair
#define Test(x)                                                                \
  int x;                                                                       \
  cin >> x
#define mem(x, val) memset((x), (val), sizeof(x));
#define sz(x) ((int)x.size())
#define sqr(x) ((x) * (x))

int main() {
  int n, ans = 0;
  cin >> n;
  string str;
  for (int i = 1; i <= n; i++) {
    cin >> str;
    if (str[0] == '+' || str[1] == '+')
      ans++;
    else
      ans--;
  }
  cout << ans << endl;
  return 0;
}