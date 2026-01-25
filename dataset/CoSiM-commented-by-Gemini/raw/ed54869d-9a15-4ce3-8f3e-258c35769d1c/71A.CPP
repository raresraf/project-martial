// Sagindyk Urazayev
// codeforces.com/problemset/problem/71/A
#include <iostream>
#include <string>

using namespace std;

int main() {
  int i, n;
  cin >> n;
  if ((n < 1) || (n > 100)) {
    cout << "Wrong value!" << endl;
  }
  string word, werd;
  for (i = 0; i < n; i++) {
    cin >> word;
    if (word.size() > 10) {
      cout << word.at(0) << word.size() - 2 << word.at(word.size() - 1) << endl;
    } else {
      cout << word << endl;
    }
  }
  return 0;
}
