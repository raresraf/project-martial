#include <bits/stdc++.h>
using namespace std;
int main() {
  int i;
  char s[100];
  scanf("%s", s);
  // cout<<s<<endl;
  int x = strlen(s);
  for (i = 0; i < x; i++) {
    s[i] = tolower(s[i]);
    if (s[i] == 'a' || s[i] == 'e' || s[i] == 'i' || s[i] == 'o' ||
        s[i] == 'u' || s[i] == 'y') {
      continue;
    } else {
      cout << "." << s[i];
    }
  }
  // cout<<s;
}
