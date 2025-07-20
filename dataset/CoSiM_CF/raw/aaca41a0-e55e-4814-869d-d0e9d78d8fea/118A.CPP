#include <cctype>
#include <iostream>
#include <sstream>
#include <string>

using std::cin;
using std::cout;
using std::isalpha;
using std::isupper;
using std::string;
using std::stringstream;
using std::tolower;

int main(void) {
  string input;
  cin >> input;
  int length = input.length();
  stringstream processed;
  for (char c : input) {
    if (c != 'A' && c != 'a' && c != 'E' && c != 'e' && c != 'I' && c != 'i' &&
        c != 'O' && c != 'o' && c != 'U' && c != 'u' && c != 'Y' && c != 'y') {
      processed.put('.');
      processed.put(tolower(c));
    }
  }
  cout << processed.str();

  return 0;
}