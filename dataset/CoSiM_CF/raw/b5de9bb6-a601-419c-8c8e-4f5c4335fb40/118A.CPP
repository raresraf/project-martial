#include <cctype>
#include <iostream>
#include <string>

using namespace std;
bool is_vowel(char);

int main() {
  string input, output;
  cin >> input;

  for (int i = 0; i < input.length(); i++) {
    if (!is_vowel(input.at(i))) {
      output.push_back('.');
      output.push_back(tolower(input.at(i)));
    }
  }

  cout << output << endl;
}

bool is_vowel(char c) {
  c = tolower(c);

  if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' || c == 'y')
    return true;
  else
    return false;
}