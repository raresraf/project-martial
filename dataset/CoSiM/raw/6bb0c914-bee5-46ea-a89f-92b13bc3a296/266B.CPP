#include <bits/stdc++.h>
#define ll long long int
#define loop(k) for (i = 0; i < k; ++i)
#define loop2(k, l) for (j = k; j < l; ++j)
#define mod 1000000007
using namespace std;
int main() {
  std::ios_base::sync_with_stdio(false); // cin.tie(NULL);
  ll n = 1, i, j, t;
  cin >> n >> t;
  string s;
  cin >> s;
  int len = s.length();
  if (len == 1)
    cout << s;
  else
    loop(t) {
      for (j = 0; j < len - 1; ++j) {
        if (s[j] == 'B' && s[j + 1] == 'G') {
          swap(s[j], s[j + 1]);
          ++j;
        }
      }
    }
  if (len != 1)
    cout << s;
  cout << "\n";
  return 0;
}