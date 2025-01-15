#include <iostream>
#include <string>

using namespace std;

int main() {
  string word;
  int word_count, word_length;

  cin >> word_count;
  for (int i = 0; i < word_count; i++) {
    cin >> word;

    word_length = word.length();
    if (word_length < 11) {
      cout << word << endl;
    } else {
      cout << word[0] << word_length - 2 << word[word_length - 1] << endl;
    }
  }

  return 0;
}