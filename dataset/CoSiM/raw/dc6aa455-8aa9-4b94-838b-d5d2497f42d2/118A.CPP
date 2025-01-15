#include <cstring>
#include <iostream>
using namespace std;

int main() {
  string input, output;
  cin >> input;

  int size;
  size = input.length();

  char letter;
  for (int i = 0; i < size; i++) {
    letter = tolower(input[i]);

    if (letter == 'a' || letter == 'e' || letter == 'y' || letter == 'i' ||
        letter == 'u' || letter == 'o') {
      continue;
    } else {
      output += '.';
      output += letter;
    }
  }
  cout << output;
  return 0;
}