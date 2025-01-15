#include <bits/stdc++.h>
using namespace std;

bool isvowel(char a) {
  if (a == 'a' || a == 'e' || a == 'i' || a == 'o' || a == 'u' || a == 'A' ||
      a == 'E' || a == 'I' || a == 'O' || a == 'U' || a == 'y' || a == 'Y')
    return true;
  else
    return false;
}

int main() {
  string t, ans = "";
  cin >> t;
  for (auto it : t) {
    if (isvowel(it))
      continue;
    else {
      ans += '.';
      ans += tolower(it);
    }
  }
  cout << ans;
}