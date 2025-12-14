#include <iostream>
#include <string>

using namespace std;

int main() {

  string word1, word2;
  cin >> word1;

  for (int i = 0; i < word1.length(); i++) {
    // "A", "O", "Y", "E", "U", "I"

    if (word1[i] == 'a' || word1[i] == 'A' || word1[i] == 'o' ||
        word1[i] == 'O' || word1[i] == 'y' || word1[i] == 'Y' ||
        word1[i] == 'E' || word1[i] == 'e' || word1[i] == 'u' ||
        word1[i] == 'U' || word1[i] == 'i' || word1[i] == 'I') {
      word1.erase(i, 1);
      i--;
    }
  }

  for (int i = 0; i < word1.length(); ++i) {
    if (isupper(word1[i]))
      word1[i] = tolower(word1[i]);
  }

  for (int j = 0; j < word1.length(); j++) {
    word2 = word2 + "." + word1[j];
  }
  cout << word2 << endl;

  return 0;
}