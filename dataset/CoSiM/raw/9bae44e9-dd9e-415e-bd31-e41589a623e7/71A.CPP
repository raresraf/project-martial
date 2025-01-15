#include <iostream>
#include <string>
#include <vector>

using namespace std;

string shortenWord(string word) {
  if (word.length() > 10) {
    return word.replace(word.begin() + 1, word.end() - 1,
                        to_string(word.length() - 2));
  }
  return word;
}

int main() {
  int noOfStrings;
  string input;
  vector<string> words;
  cin >> noOfStrings;
  for (int i = noOfStrings + 1; i > 0; i--) {
    getline(cin, input);
    words.push_back(input);
  }
  for (auto word : words) {
    cout << shortenWord(word) << endl;
  }

  return 0;
}
