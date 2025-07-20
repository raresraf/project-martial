#include <iostream>
#include <string>
using namespace std;

int main() {
  string word;
  cin >> word;
  int tam = word.length();
  for (int i = 0; i < tam; i++) {
    if (word.at(i) == 97 || word.at(i) == 101 || word.at(i) == 105 ||
        word.at(i) == 111 || word.at(i) == 117 || word.at(i) == 121) {
      word.erase(word.begin() + i);
      i--;
    }
    tam = word.length();
  }

  for (int j = 0; j < tam; j++) {
    if (word.at(j) == 65 || word.at(j) == 69 || word.at(j) == 73 ||
        word.at(j) == 79 || word.at(j) == 85 || word.at(j) == 89) {
      word.erase(word.begin() + j);
      j--;
    }
    tam = word.length();
  }

  for (int k = 0; k < tam; k++) {
    if (word.at(k) > 64 && word.at(k) < 91) {
      word.at(k) = word.at(k) + 32;
    }
    tam = word.length();
  }

  for (int l = 0; l < tam; l++) {
    if (word.at(l) != 46) {
      word.insert(l, ".");
      l++;
    }
    tam = word.length();
  }

  cout << word;
}