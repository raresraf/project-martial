#include <cctype>
#include <iostream>
using namespace std;

int main() {
  string s, result = "";
  cin >> s;
  for (int i = 0; i < s.size(); i++) {
    char temp = tolower(s[i]);
    switch (temp) {
    case 'a':
    case 'e':
    case 'i':
    case 'o':
    case 'u':
    case 'y':
      continue;
    default:
      result += ".";
      result += temp;
      // cout << result << endl;
    }
  }
  cout << result << endl;
  return 0;
}
