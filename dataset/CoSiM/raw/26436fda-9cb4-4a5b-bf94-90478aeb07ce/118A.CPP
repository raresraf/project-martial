// CF https://codeforces.com/problemset/problem/118/A

#include <bits/stdc++.h>

using namespace std;

bool isVowel(char x) {
  if (x == 'a' || x == 'e' || x == 'i' || x == 'o' || x == 'u' || x == 'y')
    return true;
  return false;
}

int32_t main() {

#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
  cout.tie(NULL);
  string s;
  cin >> s;
  string ans = "";
  for (char x : s) {
    if (!isVowel(tolower(x))) {
      ans += ".";
      ans += tolower(x);
    }
  }
  cout << ans;
  return 0;
}