#include <iostream>

using namespace std;

int main() {
  int n;
  cin >> n;
  n += 1;
  string words[n];

  for (int i = 0; i < n; i++) {
    getline(cin, words[i]);
  }

  for (int i = 0; i < n; i++) {
    string word = words[i];
    int length = word.length();
    if (length >= 11) {
      cout << word.at(0) << length - 2 << word.at(length - 1) << "\n";
    } else {
      cout << word << "\n";
    }
  }
  return 0;
}
