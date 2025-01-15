#include <bits/stdc++.h>
#include <cctype>
using namespace std;

string deleteAllVowel(string in) {
  regex pattern("[aeiouyAEIOUY]");

  return regex_replace(in, pattern, "");
}

string insertDotBeforeConsonants(string in) {
  string out = "";

  for (char c : in) {
    out += (".");
    out += c;
  }

  return out;
}

string toLowerCase(string in) {
  string out = "";

  for (auto c : in) {
    if ((int)c < 97 && (int)c != 46) {
      char temp = (char)((int)c - 65 + 97);
      out += temp;

    } else {
      out += c;
    }
  }
  return out;
}

int main() {
  string str;
  cin >> str;

  cout << toLowerCase(insertDotBeforeConsonants(deleteAllVowel(str)));

  return 0;
}
