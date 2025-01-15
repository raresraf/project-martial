#include <iostream>

using namespace std;

int main() {
  int i = 0;
  char word1[100], word2[100];
  cin >> word1 >> word2;
  while (word1[i] != '\0') {
    if (tolower(word1[i]) != tolower(word2[i])) {
      if (tolower(word1[i]) > tolower(word2[i])) {
        cout << "1";
        break;
      } else {
        cout << "-1";
        break;
      }
    }
    i++;
    if (word1[i] == '\0')
      cout << "0";
  }
  return 0;
}