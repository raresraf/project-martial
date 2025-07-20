#include <iostream>
using namespace std;

char vowels[] = {'a', 'e', 'i', 'o', 'u', 'y'};

bool Vowel(char c) {
  for (int i = 0; i < sizeof(vowels); i++) {
    if (tolower(c) == vowels[i]) {
      return true;
    }
  }
  return false;
}

int main() {

  string s;
  cin >> s;
  for (int i = 0; i < s.length(); i++) {
    s[i] = tolower(s[i]);
    if (Vowel(s[i])) {
    } else {
      cout << "." << s[i];
    }
  }
  cout << "\n";
}
