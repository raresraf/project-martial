#include <iostream>
#include <string>
using namespace std;

string toLowerCase(string before) {
  for (int i = 0; i < before.length(); i++) {
    if ((int)before[i] <= 90 && (int)before[i] >= 65) {
      before[i] += 32;
    }
  }

  return before;
}

bool isVoyel(char c) {
  if (c == 'a' || c == 'o' || c == 'y' || c == 'e' || c == 'u' || c == 'i')
    return true;
  else
    return false;
}

int main() {

  string input;
  cin >> input;

  input = toLowerCase(input);

  for (int i = 0; i < input.length(); i++) {
    if (isVoyel(input[i])) {
      continue;
    } else {
      cout << ".";
      cout << input[i];
    }
  }

  return 0;
}
