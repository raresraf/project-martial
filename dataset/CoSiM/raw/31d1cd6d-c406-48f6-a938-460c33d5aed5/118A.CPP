#include <iostream>
#include <vector>

using namespace std;
bool beVowel(char ch) {
  if (ch == 'A' || ch == 'O' || ch == 'Y' || ch == 'E' || ch == 'U' ||
      ch == 'I')
    return true;
  if (ch == 'a' || ch == 'o' || ch == 'y' || ch == 'e' || ch == 'u' ||
      ch == 'i')
    return true;
  return false;
}
int main() {
  string s;
  std::cin >> s;
  std::vector<char> v;
  for (int i = 0; i < s.length(); i++) {
    if (beVowel(s[i]) == false) {
      v.push_back('.');
      if (int(s[i]) < 97) {
        v.push_back(char(int(s[i]) + 32));
      } else
        v.push_back(s[i]);
    }
  }
  for (int j = 0; j < v.size(); j++) {
    std::cout << v[j];
  }
  return 0;
}