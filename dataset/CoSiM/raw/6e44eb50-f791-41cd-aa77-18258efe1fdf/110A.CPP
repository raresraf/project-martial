// link:
// [prajwal1721]
// g++ -Wall -Wextra -Wshadow filename.cpp
// ./a.out >input <output
/*comments:
 */

#include <bits/stdc++.h>
using namespace std;
#define mod 1000000007
#define gcd(a, b) __gcd(a, b)
#define lcm(a, b) (a*(b/gcd(a,b))
#define bits(x) __builtin_popcountll(x)
#define vi vector<int>
#define vl vector<ll>
#define all(v) v.begin(), v.end()
#define UNIQUE(X) (X).erase(unique(all(X)), (X).end())

typedef long long int ll;
void solve() {
  string s;
  cin >> s;
  // cout<<s<<" ";
  ll cnt = 0;
  for (int i = 0; i < s.length(); i++) {
    if (s[i] == '4' || s[i] == '7') {
      cnt++;
    }
  }
  s = to_string(cnt);
  // cout<<s<<" ";
  for (int i = 0; i < s.length(); i++)
    if (s[i] != '4' && s[i] != '7') {
      cout << "NO\n";
      return;
    }
  cout << "YES\n";
}

int main() {
  ios::sync_with_stdio(false);
  cin.tie(nullptr);
  // ll t;cin>>t;
  // cout<<t<<"Hi test complete\n";
  // while(t--)
  { solve(); }

  return 0;
}