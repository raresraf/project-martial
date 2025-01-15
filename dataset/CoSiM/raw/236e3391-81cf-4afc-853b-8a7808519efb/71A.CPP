/** *******************
 * Ahmed El-Taweel
 * ahmed.m.eltaweel@gmail.com
 *********************************/

#include <iostream>
#include <string>

using namespace std;

string input;
int noTest;

bool checkLength(string input) {

  if (input.length() > 10) {
    return true;
  } else {
    return false;
  }
}

// count the chars in string
int countChars(string input) { return input.length(); }

int main() {

#ifndef OLINE_JUDGE
  // freopen("c.in" , "rt" , stdin);
#endif

  cin >> noTest;

  for (int i = 0; i < noTest; i++) {
    cin >> input;

    if (checkLength(input)) {
      // implement the algorithm
      cout << input.front() << countChars(input) - 2 << input.back() << endl;

    } else {
      cout << input << endl;
    }
  }

  return 0;
}
