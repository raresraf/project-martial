// CodeForces
// 118A - StringTask

#include <iostream>
using namespace std;

bool isVowel(char c) {
  c = (char)tolower(c);

  if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' || c == 'y')
    return true;
  else
    return false;
}

int main() {
  string word;
  cin >> word;

  string output("");

  for (unsigned int i = 0; i < word.size(); i++) {
    if (!isVowel(word[i])) {
      output = output + "." + (char)tolower(word[i]);
    }
  }

  cout << output;

  return 0;
}
