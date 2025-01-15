#include <iostream>

using namespace std;
int main() {
  int n, wordLength;
  cin >> n;

  string wordsList[101];
  for (int i = 0; i < n; i++) {
    cin >> wordsList[i];
  }

  for (int i = 0; i < n; i++) {
    wordLength = wordsList[i].length();
    if (wordLength <= 10) {
      cout << wordsList[i] << endl;
    } else
      cout << (wordsList[i])[0] << wordLength - 2
           << (wordsList[i])[wordLength - 1] << endl;
  }
}
