#include <bits/stdc++.h>
using namespace std;
#pragma GCC optimize("-O3")

#define ll long long int
#define pll pair<ll, ll>
#define pls pair<ll, string>
#define psl pair<string, ll>
#define plc pair<ll, char>
#define pcl pair<char, ll>
#define pss pair<string, string>
#define maxheap(type) priority_queue<type>
#define minheap(type) priority_queue<type, vector<type>, greater<type>>
#define setbits(x) __builtin_popcountll(x)
#define zerobits(x) __builtin_ctzll(x)
#define mod 1000000007
#define inf 1e18
#define precise(x, y) fixed << setprecision(y) << x
#define tol(s) transform(s.begin(), s.end(), s.begin(), ::tolower);
#define tou(s) transform(s.begin(), s.end(), s.begin(), ::toupper);
#define rep(n) for (auto i = 0; i < n; i++)

int main() {
  ios_base::sync_with_stdio(0);
  cin.tie(NULL);
  cout.tie(NULL);
  string s1, s2;
  cin >> s1 >> s2;
  if (s1.size() != s2.size()) {
    cout << "NO\n";
    return 0;
  }
  ll n = s1.size();
  vector<ll> st;
  rep(n) {
    auto ch1 = s1[i], ch2 = s2[i];
    if (ch1 != ch2) {
      st.push_back(i);
    }
    if (st.size() > 2) {
      cout << "NO\n";
      return 0;
    }
  }
  if ((st.size() != 2) ||
      !((s1[st[0]] == s2[st[1]]) && (s1[st[1]] == s2[st[0]]))) {
    cout << "NO\n";
    return 0;
  }
  cout << "YES\n";
  return 0;
}