#include <bits/stdc++.h>
#define ll long long int
#define loop(k) for (i = 0; i < k; ++i)
#define loop2(k, l) for (j = k; j < l; ++j)
#define mod 1000000007
using namespace std;
int main() {
  std::ios_base::sync_with_stdio(false); // cin.tie(NULL);
  ll n = 1, i, j, t;
  cin >> n;
  string s;
  cin >> s;
  int count = 0;
  loop(n - 1) {
    if (s[i] == s[i + 1])
      count++;
  }
  cout << count;
  cout << "\n";
  return 0;
}