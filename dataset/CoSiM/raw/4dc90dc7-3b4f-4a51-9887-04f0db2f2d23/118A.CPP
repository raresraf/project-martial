#include <iostream>
using namespace std;
int main(int argc, char const *argv[]) {
  string s;
  cin >> s;
  string ans;
  char a;
  for (int i = 0; i < s.size(); i++)
    if (s[i] == 'a' || s[i] == 'A' || s[i] == 'e' || s[i] == 'E' ||
        s[i] == 'i' || s[i] == 'I' || s[i] == 'o' || s[i] == 'O' ||
        s[i] == 'u' || s[i] == 'U' || s[i] == 'y' || s[i] == 'Y')
      continue;
    else {
      if (s[i] >= 97 && s[i] < 123) {
        cout << '.' << s[i];
      } else if (s[i] >= 65 && s[i] < 91) {
        a = s[i] + 32;
        cout << '.' << a;
      }
    }
  return 0;
}