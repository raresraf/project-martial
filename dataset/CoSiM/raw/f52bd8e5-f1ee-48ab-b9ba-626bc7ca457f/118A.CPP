#include <bits/stdc++.h>
using namespace std;
int main() {
  string s, a = "";
  cin >> s;
  for (int i = 0; i < s.size(); i++) {
    if (s[i] >= 'A' && s[i] <= 'Z')
      s[i] = s[i] - 'A' + 'a';
    if (s[i] != 'a' && s[i] != 'e' && s[i] != 'i' && s[i] != 'o' &&
        s[i] != 'u' && s[i] != 'y')
      a += s[i];
  }
  for (int i = 0; i < a.size(); i++)
    cout << "." << a[i];
  printf("\n");
  return 0;
}
