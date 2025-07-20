#include <cstring>
#include <iostream>
#include <string.h>

using namespace std;

main() {
  string word;
  cin >> word;
  int n = word.length();
  int i, k = 0;
  for (i = 0; i < n; i++) {
    if (word[i] < 97)
      word[i] = word[i] + 32;
    if (word[i] != 97 && word[i] != 101 && word[i] != 105 && word[i] != 111 &&
        word[i] != 117 && word[i] != 121) {
      cout << "." << word[i];
    }
  }
}
