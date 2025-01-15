// Problem Link : https://codeforces.com/problemset/problem/71/A

#include <bits/stdc++.h>
#define ll long long

using namespace std;

void solve() {
  ll n;
  cin >> n;
  string str;
  string temp;
  ll len;
  for (ll i = 0; i < n; i++) {
    cin >> str;
    len = str.length();
    if (len <= 10) {
      cout << str << endl;
    } else {
      temp = str[0] + to_string(len - 2) + str[len - 1];
      cout << temp << endl;
    }
  }
}

int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

#ifndef ONLINE_JUDGE
  // for getting input from input.txt
  freopen("input.txt", "r", stdin);
  // for writing output to output.txt
  freopen("output.txt", "w", stdout);
#endif

  solve();

  return 0;
}
