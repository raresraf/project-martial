#include <bits/stdc++.h>

using namespace std;

#define deb(x) cout << #x << " = " << x << '\n'
#define ll long long
#define PI 3.1415926535897932384626

void crap();

int main() {
  crap();

  string s;
  cin >> s;
  sort(s.begin(), s.end());
  // deb(s);
  int c = 0;

  for (int i = 0; i < s.length() - 1; ++i) {
    if (s[i] != s[i + 1]) {
      ++c;
    }
  }
  if (c % 2 == 0) {
    cout << "IGNORE HIM!";
  } else {
    cout << "CHAT WITH HER!";
  }
  return 0;
}

void crap() { ios_base::sync_with_stdio(false), cin.tie(0), cout.tie(0); }