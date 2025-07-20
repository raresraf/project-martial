#include <bits/stdc++.h>
#define ll long long
#define vi vector<int>
using namespace std;
void solve() {
  int a;
  cin >> a;
  string s;
  cin >> s;
  int count = 0;
  for (int i = 0; i < s.length(); i++) {
    while (s[i] == s[i + 1]) {
      count++;
      i++;
    }
  }
  cout << count;
}
int main() {
  // int t;
  // cin>>t;
  solve();
}
