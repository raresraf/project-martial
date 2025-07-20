#include <bits/stdc++.h>
#define LOG(x) cout << x << "\n"
// -std=c++11
using namespace std;

bool isVowel(char x) {
  // The problem statement includes 'y' as a vowel
  vector<char> vowels = {'a', 'e', 'i', 'o', 'u', 'y'};
  for (char c : vowels) {
    // 32 is diff b/w ASCII 'A' and 'a'
    if ((x == c) || (x == c - 32)) {
      return true;
    }
  }
  return false;
}

int main() {
  string s;
  cin >> s;
  int len = s.size();
  string out = "";
  for (int i = 0; i < len; i++) {
    if (!isVowel(s[i])) {
      // Insert "." before each consonant
      out.push_back('.');
      // Substitute uppercase consonants with lowercase
      out.push_back(tolower(s[i]));
    }
  }
  LOG(out);
  return 0;
}