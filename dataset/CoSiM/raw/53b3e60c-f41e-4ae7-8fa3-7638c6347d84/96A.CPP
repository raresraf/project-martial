#include <bits/stdc++.h>
using namespace std;
#define ll long long int
#define YES cout << "YES";
#define NO cout << "NO";
int main() {
  string s;
  cin >> s;
  int ans = 1;
  int ok = 0;
  for (int i = 1; i < s.length(); i++) {
    if (s[i] == s[i - 1])
      ans++;
    else
      ans = 1;
    if (ans == 7) {
      ok = 1;
      break;
    }
  }
  if (ok)
    YES else NO
}
