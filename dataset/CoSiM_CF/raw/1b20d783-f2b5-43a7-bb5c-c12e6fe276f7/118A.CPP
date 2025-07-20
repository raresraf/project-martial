#include <bits/stdc++.h>
using namespace std;

int main() {

  string s;
  cin >> s;
  int l = s.size();
  for (int i = 0; i < l; i++) {
    if (s[i] == 'a' || s[i] == 'e' || s[i] == 'i' || s[i] == 'o' ||
        s[i] == 'u' || s[i] == 'y' || s[i] == 'A' || s[i] == 'E' ||
        s[i] == 'I' || s[i] == 'O' || s[i] == 'U' || s[i] == 'Y') {
      continue;
    } else {

      printf(".%c", tolower(s[i]));
    }
  }

  printf("\n");
  return 0;
}
