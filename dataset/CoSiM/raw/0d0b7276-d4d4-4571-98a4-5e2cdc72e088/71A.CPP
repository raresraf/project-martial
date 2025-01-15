#include <iostream>
#include <string>
#include <vector>
using namespace std;

int main() {
  // the number of lines
  int n = 0;
  cin >> n;
  vector<string> words;
  string word = "";
  int stringSize = 0;
  int stringLongSize = 0;
  for (int i = 0; i < n; i++) {
    cin >> word;
    words.push_back(word);
  }

  for (int i = 0; i < n; i++) {
    stringSize = words[i].size();
    if (stringSize > 10) {
      stringLongSize = stringSize - 2;
      char x = words[i][0];
      char y = words[i][stringSize - 1];
      cout << x << stringLongSize << y << endl;
    } else {
      cout << words[i] << endl;
    }
  }

  return 0;
}
