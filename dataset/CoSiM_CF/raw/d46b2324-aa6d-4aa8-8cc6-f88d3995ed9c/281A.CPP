#include <bits/stdc++.h>
using namespace std;

int main() {
  string s;
  cin >> s;
  int len = s.length();
  char c = toupper(s[0]);
  printf("%c", c);
  int i;
  for (i = 1; i < len; i++) {
    cout << s[i];
  }
  printf("\n");
  return 0;
}
