#include <iostream>
#include <string>
using namespace std;
bool substring(string s);
int main() {
  string input;
  getline(cin, input);
  if (substring(input) == false)
    cout << "YES\n";
  else if (substring(input) == true)
    cout << "NO\n";
  return 0;
}
bool substring(string s) {
  int i, j, cnt = 0;
  string s1 = "hello";
  int len1 = s.length(), len2 = s1.length();
  for (i = 0; i <= (len1 - len2); i++) {
    cnt = 0;
    if (s[i] != s1[0])
      continue;
    for (j = 0; j < len1; j++) {
      if (s[i + j] == s1[cnt])
        cnt++;
      if (cnt == 5)
        return false;
    }
  }
  return true;
}
