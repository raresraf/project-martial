#include <bits/stdc++.h>

using namespace std;

#define deb(x) cout << #x << " = " << x << '\n'
#define ll long long
#define PI 3.1415926535897932384626

void crap();
void solve();

int main() {
  crap();
  solve();
  return 0;
}

void solve() {
  string s;
  cin >> s;
  vector<int> num;
  for (int i = 0; i < s.length(); i += 2) {
    num.push_back(s[i] - 48);
  }

  sort(num.begin(), num.end());
  string s2 = "";

  for (int i = 0; i < num.size(); ++i) {
    if (i == num.size() - 1) {
      s2 += to_string(num[i]);
    } else {
      s2 += to_string(num[i]) + "+";
    }
  }

  for (int i = 0; i < s2.size(); ++i) {
    cout << s2[i];
  }
}

void crap() { ios_base::sync_with_stdio(false), cin.tie(0), cout.tie(0); }