#include <bits/stdc++.h>
using namespace std;
int main() {

  char s[1003];
  cin >> s;
  s[0] = toupper(s[0]);
  for (int i = 0; i < strlen(s); i++)
    cout << s[i];
  return 0;
}
