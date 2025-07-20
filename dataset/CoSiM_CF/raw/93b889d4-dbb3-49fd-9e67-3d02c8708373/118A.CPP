#include <iostream>
#include <locale>

using namespace std;

int main() {
  char word[100];
  cin >> word;
  char vowels[12] = {'a', 'o', 'y', 'e', 'u', 'i',
                     'A', 'O', 'Y', 'E', 'U', 'I'};

  int i = 0;
  while (word[i]) {
    bool isVowel = false;
    for (int j = 0; j < 12; j++) {
      if (word[i] == vowels[j]) {
        isVowel = true;
        i++;
        break;
      }
    }

    if (isVowel)
      continue;
    cout << '.' << (char)tolower(word[i]);
    i++;
  }
  return 0;
}
