#include <bits/stdc++.h>
#define ll long long int
#define loop(k) for (i = 0; i < k; ++i)
#define loop2(k, l) for (j = k; j < l; ++j)
#define mod 1000000007
using namespace std;
int main() {
  std::ios_base::sync_with_stdio(false);
  cin.tie(NULL);
  ll t = 1, i = 0, j = 0, k = 0;
  cin >> t;
  while (t--) {
    string s;
    cin >> s;
    int len = s.length();
    if (len > 10) {
      cout << s[0] << len - 2 << s[len - 1];
    } else
      cout << s;
    cout << "\n";
  }
  return 0;
}