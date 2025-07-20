#include <bits/stdc++.h>
#include <string.h>
using namespace std;
int main() {
  ios::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);
  string str;
  string cpy;
  getline(cin, str);
  for (int i = 0; i < str.length(); i++) {
    if (str[i] == 'a' || str[i] == 'e' || str[i] == 'i' || str[i] == 'o' ||
        str[i] == 'u' || str[i] == 'y' || str[i] == 'A' || str[i] == 'E' ||
        str[i] == 'O' || str[i] == 'I' || str[i] == 'U' || str[i] == 'Y')
      continue;
    else {
      cpy += '.';
      cpy += towlower(str[i]);
    }
  }

  cout << cpy;

  return 0;
}
