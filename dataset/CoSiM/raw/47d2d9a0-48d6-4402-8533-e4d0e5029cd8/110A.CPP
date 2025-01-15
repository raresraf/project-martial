#include <bits/stdc++.h>
#define ll long long
#define vi vector<int>
using namespace std;
void solve() {
  ll a;
  cin >> a;
  string b;
  b = to_string(a);

  int count = 0, i = 0, y = b.length();
  for (i = 0; i < y; i++) {
    if (b[i] == '4' || b[i] == '7')
      count++;
  }
  if (count == 4 || count == 7)
    cout << "YES\n";
  else
    cout << "NO\n";
}
int main() {
  // int t;
  // cin>>t;
  solve();
}
