#include <bits/stdc++.h>
using namespace std;

set<char> vowels{'A', 'E', 'I', 'O', 'U', 'Y'};

bool isUpper(char c) { return c >= 'A' && c <= 'Z'; }
bool isLower(char c) { return c >= 'a' && c <= 'z'; }

char toLower(char c) {
  if (isUpper(c))
    return c - 'A' + 'a';
  return c;
}

char toUpper(char c) {
  if (isLower(c))
    return c - 'a' + 'A';
  return c;
}
bool isVowel(char c) {
  if (isLower(c))
    c = toUpper(c);
  return vowels.count(c) > 0;
}

int main(int argc, char const *argv[]) {
  char c;
  while (cin >> c) {

    if (!isVowel(c)) {
      cout << "." << toLower(c);
    }
  }
  return 0;
}