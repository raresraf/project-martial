#include <iostream>
#include <string>

using namespace std;

int main() {
  string etalon = "hello", word;
  cin >> word;
  size_t j = 0;
  for (size_t i = 0; i < word.size() && j < etalon.size(); ++i) {
    if (word[i] == etalon[j])
      ++j;
  }
  string answer = ((etalon.size() == j) ? "YES" : "NO");
  cout << answer << endl;
  return 0;
}
