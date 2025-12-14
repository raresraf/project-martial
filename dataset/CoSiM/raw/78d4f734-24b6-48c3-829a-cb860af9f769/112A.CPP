#include <iostream>

using namespace std;

int main() {
  string word1, word2;
  int result;

  getline(cin, word1);
  getline(cin, word2);

  for (auto it = word1.begin(); it != word1.end(); it++) {
    if (*it < 97)
      *it = *it + 32;
  }
  for (auto it = word2.begin(); it != word2.end(); it++) {
    if (*it < 97)
      *it = *it + 32;
  }

  result = word1.compare(word2);
  if (result >= 0)
    cout << result << endl;
  else
    cout << -1 << endl;

  return 0;
}
