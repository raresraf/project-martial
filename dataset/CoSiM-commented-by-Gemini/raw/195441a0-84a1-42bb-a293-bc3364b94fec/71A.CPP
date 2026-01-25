#include <iostream>
#include <string>

using namespace std;

int main() {
  int n;
  cin >> n;
  for (int i = 0; i < n; i++) {
    string word;
    cin >> word;
    if (word.size() <= 10) {
      cout << word << endl;
    } else {
      int n = word.size() - 2;
      string s = to_string(n);
      string o = word[0] + s + word[word.size() - 1];
      cout << o << endl;
    }
  }
  return 0;
}
