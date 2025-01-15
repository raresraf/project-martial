#include <bits/stdc++.h>
using namespace std;

int main() {
  string word;
  cin >> word;
  vector<char> vowels{'a', 'e', 'i', 'o', 'u', 'y'};
  string newWord = "";
  for (int i = 0; i < (int)word.size(); i++) {
    if (newWord[newWord.size() - 1] != '.') {
      newWord += ".";
    }
    if (find(vowels.begin(), vowels.end(), tolower(word[i])) == vowels.end()) {
      newWord += tolower(word[i]);
    }
  }
  if (newWord[newWord.size() - 1] == '.') {
    newWord = newWord.substr(0, newWord.size() - 1);
  }
  cout << newWord;
}
