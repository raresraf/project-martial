#include <iostream>
#include <string>

using namespace std;

char convertToLowerCase(char ch) {
  if (ch - 'a' < 0)
    return ch + 32;
  return ch;
}

bool isVowel(char ch) {
  ch = convertToLowerCase(ch);
  if (ch == 'a' || ch == 'o' || ch == 'y' || ch == 'e' || ch == 'u' ||
      ch == 'i')
    return true;
  return false;
}

int main() {
  string s, updated_string = "";
  cin >> s;
  for (char ch : s) {
    if (isVowel(ch))
      continue;
    updated_string += "." + string(1, convertToLowerCase(ch));
  }
  cout << updated_string << endl;
  return 0;
}
