#include <iostream>
using namespace std;
int main() {
  string input;
  char c;
  cin >> input;
  for (int i = 0; i < input.length(); i++) {
    if (input[i] == 'a' || input[i] == 'A' || input[i] == 'e' ||
        input[i] == 'E' || input[i] == 'i' || input[i] == 'I' ||
        input[i] == 'o' || input[i] == 'O' || input[i] == 'u' ||
        input[i] == 'U' || input[i] == 'y' || input[i] == 'Y') {
      continue;
    } else {
      cout << ".";
      if (input[i] >= 65 && input[i] <= 90) {
        c = input[i] + 32;
        cout << c;
      } else {
        cout << input[i];
      }
    }
  }
}
