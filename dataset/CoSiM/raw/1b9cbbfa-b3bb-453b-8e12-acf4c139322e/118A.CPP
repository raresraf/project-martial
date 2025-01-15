#include <cctype>
#include <iostream>

using namespace std;

inline bool isVowel(char c) {
  return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' || c == 'y';
}

int main() {
  string s, aux;
  cin >> s;
  for (int i = 0; i < s.size(); ++i) {
    if (!isVowel(tolower(s[i]))) {
      aux.push_back('.');
      aux.push_back(tolower(s[i]));
    }
  }
  cout << aux;
}