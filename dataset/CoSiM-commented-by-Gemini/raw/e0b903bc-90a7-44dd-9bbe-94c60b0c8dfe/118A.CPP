#include <bits/stdc++.h>

using namespace std;

#define lli long long int

lli mod = 1000000007;

bool isVowel(char c) {

  char lc = tolower(c);

  if (lc == 'a' || lc == 'e' || lc == 'i' || lc == 'o' || lc == 'y' ||
      lc == 'u') {

    return true;
  }

  return false;
}

void task() {

  string str;
  cin >> str;

  string newString = "";

  for (int i = 0; i < str.length(); i++) {
    if (!isVowel(str[i])) {
      newString.push_back('.');
      newString.push_back(tolower(str[i]));
    }
  }

  cout << newString << '\n';
}

int main() { task(); }