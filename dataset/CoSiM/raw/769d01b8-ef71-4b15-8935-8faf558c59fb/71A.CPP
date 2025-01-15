#include <iostream>

using namespace std;

int main() {
  int n;
  string word, abbreviation;
  cin >> n;
  cin.get();
  for (int i = 0; i < n; i++) {
    getline(cin, word);
    if (word.size() > 10) {
      abbreviation.push_back(word[0]);
      abbreviation += std::to_string(word.size() - 2);
      abbreviation.push_back(word[word.size() - 1]);
      cout << abbreviation << endl;
      abbreviation.clear();
    } else
      cout << word << endl;
  }

  return 0;
}
