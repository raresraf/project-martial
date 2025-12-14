#include <iostream>
#include <string>

using namespace std;

char tolower(char ch) {
  if (ch >= 'A' && ch <= 'Z')

  {
    return ch - ('Z' - 'z');
  }
  return ch;
}

int main() {
  string s1, s2;
  int flag = 0, i;
  cin >> s1;
  cin >> s2;
  for (i = 0; i < s1.length(); i++) {
    s1[i] = tolower(s1[i]);
    s2[i] = tolower(s2[i]);
    if (s1[i] != s2[i]) {
      flag = 1;
      break;
    }
  }
  if (flag == 0)
    cout << "0";
  else {
    if (s1[i] < s2[i])
      cout << "-1";
    else
      cout << "1";
  }
  return 0;
}
