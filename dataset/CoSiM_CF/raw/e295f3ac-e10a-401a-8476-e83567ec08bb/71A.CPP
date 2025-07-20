#include <iostream>
#include <string>
using namespace std;

int main() {
  string word;
  int i, n;

  cin >> n;
  for (i = 0; i < n; i++) {
    cin >> word;
    int wordlen = word.length();
    if (wordlen <= 10) {
      cout << word << endl;
    } else {
      cout << word[0] << wordlen - 2 << word[wordlen - 1] << endl;
    }
  }
}
