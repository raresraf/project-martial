#include <iostream>
#include <string>

using namespace std;

int main() {
  char vowels[12] = {'a', 'o', 'y', 'e', 'u', 'i',
                     'A', 'O', 'Y', 'E', 'U', 'I'};
  string input;
  cin >> input;
  for (int i = 0; i < input.length(); i++) {
    for (int j = 0; j < 12; j++) {
      char temp = input.at(i);
      char vowel = vowels[j];
      if (temp == vowel) {
        input.erase(i, 1);
        i--;
        break;
      }
    }
  }
  for (int i = 0; i < input.length(); i++) {
    for (int j = 0; j < 12; j++) {
      char temp = input.at(i);
      char vowel = vowels[j];
      if (temp == vowel) {
        input.erase(i, 1);
        break;
      }
    }
    char temp = input.at(i);
    if (temp >= 65 && temp < 97) {
      temp = temp + 32;
      cout << "." << temp;
    } else {
      cout << "." << temp;
    }
  }
  return 0;
}