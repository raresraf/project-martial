#include <algorithm>
#include <iostream>
#include <string>

using namespace std;

char consonants[6] = {'a', 'e', 'i', 'o', 'u', 'y'};

bool isConsonant(char a) {
  for (char c : consonants) {
    if (a == c)
      return false;
  }
  return true;
}

int main() {

  string input;

  cin >> input;

  for_each(input.begin(), input.end(), [](char &c) {
    c = ::tolower(c);
    if (isConsonant(c)) {
      cout << '.' << c;
    }
  });

  cout << endl;

  return 0;
}
