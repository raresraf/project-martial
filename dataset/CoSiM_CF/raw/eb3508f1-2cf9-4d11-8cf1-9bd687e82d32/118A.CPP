
/*                     */

#include <bits/stdc++.h>
using namespace std;

int main() {
  string s;

  cin >> s;

  for (int i = 0; i < s.size(); i++) {
    if (tolower(s[i]) == 'a' || tolower(s[i]) == 'e' || tolower(s[i]) == 'i' ||
        tolower(s[i]) == 'o' || tolower(s[i]) == 'u' || tolower(s[i]) == 'y')
      s[i] = '/0';
    else {
      s[i] = tolower(s[i]);
      cout << '.' << s[i];
    }
  }
  return 0;
}
