#include <algorithm>
#include <cctype>
#include <iostream>
#include <string>

using namespace std;

int main() {
  string word, newWord;
  cin >> word;

  char vowels[] = {'A', 'E', 'I', 'O', 'U', 'Y', 'a', 'e', 'i', 'o', 'u', 'y'};

  for (int i = 0; i < word.size(); ++i) {
    if (find(vowels, vowels + 12, word[i]) == vowels + 12) {
      newWord.push_back('.');
      newWord.push_back(tolower(word[i]));
    }
  }

  cout << newWord;
  return 0;
}
