// CodeForces
// Way too long words
// GNU C++17

#include <iostream>
using namespace std;

int main() {
  int numberOfWords;
  cin >> numberOfWords;
  while (numberOfWords > 0) {
    string currentWord;
    cin >> currentWord;
    if (currentWord.length() <= 10) {
      cout << currentWord << endl;
    } else {
      int count = 0;
      for (int i = 1; i < currentWord.length() - 1; i++) {
        count++;
      }
      cout << currentWord[0] << count << currentWord[currentWord.length() - 1]
           << endl;
    }
    numberOfWords--;
  }
}
